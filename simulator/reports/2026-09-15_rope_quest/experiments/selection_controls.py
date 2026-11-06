"""Offline chunk selection with captured queries and prefill-only exact mass.
No generation. Native chunks versus fixed-16 partition. Whole-chunk greedy
admission skips items that do not fit B=2048, no pinned tokens. Common RoPE
shifts preserve all logits. Chunk-relative collapse changes the target too.
"""
from pathlib import Path
import json,csv,math,time,hashlib
import numpy as np
import torch
from real_kv_interventions import rotate,digest

ROOT=Path('/workspace/rope_quest_audit_20260915_codex')
OUT=ROOT/'selection'; OUT.mkdir(exist_ok=True)
torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
torch.set_float32_matmul_precision('highest')

def ranks(v):
    order=np.argsort(v,kind='stable'); out=np.empty(len(v),float)
    sv=v[order]; a=0
    while a<len(v):
        b=a+1
        while b<len(v) and sv[b]==sv[a]: b+=1
        out[order[a:b]]=(a+b-1)/2; a=b
    return out

def corr(a,b):
    if np.std(a)==0 or np.std(b)==0:return float('nan')
    return float(np.corrcoef(ranks(a),ranks(b))[0,1])

def select(score,length,budget=2048):
    selected=np.zeros(len(score),bool); used=0
    for i in np.argsort(-score,kind='stable'):
        if used+length[i]<=budget:selected[i]=True;used+=int(length[i])
    return selected,used

@torch.inference_mode()
def run():
    manifest=json.loads((ROOT/'full'/'manifest.json').read_text()); rows=[]
    theta=5_000_000.0**(-torch.arange(64,device='cuda')/64)
    begin=time.time()
    invariance=[]; metadata=[]
    for cap in manifest['captures']:
        p=Path('/workspace/scoring_case_study')/('dumps_'+cap['capture'].split('/')[0])/cap['capture'].split('/')[1]
        T=cap['n_tokens']; ch=json.loads((p/'chunks.json').read_text())['chunks']
        assert ch[0][0]==0 and ch[-1][1]==T-1
        assert all(ch[j][1]+1==ch[j+1][0] for j in range(len(ch)-1))
        metadata.append(dict(capture=cap['capture'],chunk_sha256=digest(p/'chunks.json'),n_chunks=len(ch)))
        for layer in [0,12,24,36]:
            k=torch.load(p/f'k_layer_{layer:02d}.pt',map_location='cpu',weights_only=True).to('cuda',torch.float32)
            qall=np.fromfile(p/f'q_layer_{layer:02d}.bin',dtype=np.float16).reshape(-1,32,128)
            calls=[0,4,len(qall)-1]
            q=torch.from_numpy(qall[calls].copy()).to('cuda',torch.float32).reshape(3,4,8,128)
            for partition,chunks in [('native',ch),('fixed16',[[s,min(s+15,T-1)] for s in range(0,T,16)])]:
                length=np.array([e-s+1 for s,e in chunks]); C=len(chunks)
                chunkids=np.repeat(np.arange(C),length); starts=np.array([s for s,e in chunks])
                ids=torch.as_tensor(chunkids,device='cuda')
                idx=ids[:,None,None].expand(-1,4,128)
                original_logits=None;original_mass=None;selected_original={}
                for mode in ['original','collapse_all','common_shift_137','common_shift_997','common_shift_4093']:
                    qm=q
                    if mode=='original':x=k
                    elif mode=='collapse_all':
                        rel=torch.as_tensor(np.arange(T)-starts[chunkids],device='cuda',dtype=torch.float32)
                        x=rotate(k,-rel[:,None,None]*theta)
                    else:
                        angle=int(mode.rsplit('_',1)[1])*theta
                        x=rotate(k,angle);qm=rotate(q,angle)
                    lo=torch.full((C,4,128),float('inf'),device='cuda')
                    hi=torch.full((C,4,128),-float('inf'),device='cuda')
                    lo.scatter_reduce_(0,idx,x,reduce='amin',include_self=True)
                    hi.scatter_reduce_(0,idx,x,reduce='amax',include_self=True)
                    qp=qm.mean(2)
                    c=(torch.einsum('qhd,chd->qch',qp,(lo+hi)/2)/math.sqrt(128)).mean(-1)
                    w=(torch.einsum('qhd,chd->qch',qp.abs(),(hi-lo)/2)/math.sqrt(128)).mean(-1)
                    logits=torch.einsum('qhgd,thd->qhgt',qm,x)/math.sqrt(128)
                    token_mass=logits.flatten(1,2).softmax(-1).mean(1)
                    mass=torch.zeros((3,C),device='cuda').scatter_add_(1,ids[None,:].expand(3,-1),token_mass)
                    err=0.;merr=0.
                    if mode=='original':original_logits=logits;original_mass=mass
                    elif mode.startswith('common'):
                        err=(logits-original_logits).abs().max().item()
                        merr=(mass-original_mass).abs().max().item()
                        assert err<.002 and merr<.0001
                        invariance.append(dict(capture=cap['capture'],layer=layer,partition=partition,mode=mode,logit_error=err,mass_error=merr))
                    c,w,mass=c.cpu().numpy(),w.cpu().numpy(),mass.cpu().numpy()
                    for qi,call in enumerate(calls):
                        density=mass[qi]/length
                        for method,score in [('quest',c[qi]+w[qi]),('width_alpha025_anchor16',c[qi]+(16/length)**.25*w[qi]),('exact_density_greedy',density)]:
                            selected,used=select(score,length)
                            if mode=='original':selected_original[qi,method]=selected.copy()
                            old=selected_original[qi,method]
                            intersection=length[selected & old].sum();union=length[selected | old].sum()
                            rows.append(dict(capture=cap['capture'],prompt_hash=cap['prompt_hash'],task=cap['task'],layer=layer,call=call,partition=partition,mode=mode,method=method,
                                n_chunks=C,budget=2048,used_tokens=used,retained_mass=mass[qi,selected].sum(),
                                corr_score_density=corr(score,density),corr_score_length=corr(score,length),
                                selected_mean_chunk_length=length[selected].mean(),selected_long_token_fraction=length[selected & (length>=128)].sum()/used,
                                selected_token_jaccard_vs_original=intersection/union,logit_invariance_max_abs_error=err,mass_invariance_max_abs_error=merr))
            print(cap['capture'],'selection layer',layer,'elapsed',round(time.time()-begin,1),flush=True)
        with (OUT/'selection_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'manifest.json').write_text(json.dumps(dict(protocol=__doc__,source_manifest_sha256=digest(ROOT/'full'/'manifest.json'),script_sha256=digest(__file__),captures=metadata,invariance=invariance,elapsed_seconds=time.time()-begin),indent=2))
    print('COMPLETE',round(time.time()-begin,1),flush=True)

if __name__=='__main__':run()
