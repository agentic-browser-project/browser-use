"""Batched all-layer action replay using the previously audited BrowSA adapter.

Only new experiment paths and orchestration are introduced. Numerical kernels,
packing, sparse attention, and parsing are imported from the pinned old run.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3

ROOT = Path('/workspace/calibration32_20261112')
OLD = Path('/workspace/semantic_ranking_grid_20261111')
MODEL = '/models/Qwen3-32B'
sys.path.insert(0, str(OLD / 'experiments'))
import runtime
runtime.ROOT = ROOT
import replay_actions
from replay_actions import ReplayAttention, action_info
from replay_grouped import make_graph
from bsa_sglang.packing_gpu import bounded_layout

CONFIGS = runtime.CONFIGS
VARIANTS = [dict(cap=cap, A=a, alpha=b) for cap in (16, 32) for a, b in CONFIGS] + [dict(cap=0, A=0, alpha=0)]
ORIGINAL_ATTENTION = modeling_qwen3.ALL_ATTENTION_FUNCTIONS['sdpa']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def prepare(tokenizer):
    source = ROOT/'inputs/candidate_messages.json'
    rows = json.loads(source.read_text())['records']
    records = []
    for row in rows:
        ids = tokenizer.apply_chat_template(row['messages'], tokenize=True,
                    add_generation_prompt=True, enable_thinking=True)
        assert 1 < len(ids) < 32768-1024, (row['task'], len(ids))
        record = {k:v for k,v in row.items() if k!='messages'}
        record.update(capture=f"webvoyager_new/req_{row['request']:03d}", input_ids=ids,
            prompt_tokens=len(ids), prompt_hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            template_options={'add_generation_prompt':True,'enable_thinking':True},geometry={})
        # Match the historical replay's open assistant header exactly.
        assert ids[-3:]==[151644,77091,198], (row['task'],ids[-10:])
        for cap in (16,32):
            spans = runtime.chunk_spans(ids, tokenizer, cap)
            lengths = spans[:,1]-spans[:,0]+1
            record['geometry'][str(cap)] = dict(spans=spans.tolist(), chunks=len(spans),
                minimum=int(lengths.min()), maximum=int(lengths.max()),
                below_merge_target=int((lengths<6).sum()))
        records.append(record)
    assert len(records)==64 and len({r['prompt_hash'] for r in records})==64
    target = ROOT/'inputs/tokenized.json'
    if target.exists():
        assert records == json.loads(target.read_text())
    else:
        atomic_json(target, records)
    atomic_json(ROOT/'geometry.json', [{k:v for k,v in r.items() if k!='input_ids'} for r in records])
    return records


def capture_prefix(model, record):
    directory = ROOT/'captures'/record['capture']
    done = directory/'capture.json'
    if done.exists():
        meta = json.loads(done.read_text())
        assert meta['prompt_hash']==record['prompt_hash'] and meta['complete']
        return meta
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    def hook(module, query, key, value, mask, **kwargs):
        data = dict(k=key[0].transpose(0,1).cpu().contiguous(),
                    v=value[0].transpose(0,1).cpu().contiguous())
        assert data['k'].shape == data['v'].shape == (record['prompt_tokens'],8,128)
        path = directory/f'layer_{module.layer_idx:02d}.pt'
        temp = path.with_suffix('.tmp')
        torch.save(data,temp)
        temp.replace(path)
        files.append(dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size))
        return ORIGINAL_ATTENTION(module,query,key,value,mask,**kwargs)
    modeling_qwen3.ALL_ATTENTION_FUNCTIONS['sdpa'] = hook
    start = time.monotonic()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        result = model(input_ids=torch.tensor([record['input_ids']],device='cuda'),
            attention_mask={'full_attention':None},use_cache=False,logits_to_keep=1)
    logits = result.logits[0,0].cpu().contiguous()
    first = int(logits.argmax())
    torch.save(logits,directory/'first_logits.pt')
    meta = dict(complete=True,task=record['task'],prompt_hash=record['prompt_hash'],
        first_token=first,files=files,elapsed_seconds=time.monotonic()-start,
        model='Qwen/Qwen3-32B',dtype='bfloat16',layers=64)
    assert len(files)==64
    atomic_json(done,meta)
    modeling_qwen3.ALL_ATTENTION_FUNCTIONS['sdpa'] = ORIGINAL_ATTENTION
    del result,logits
    gc.collect();torch.cuda.empty_cache()
    print('PREFILL',record['task'],record['prompt_tokens'],round(meta['elapsed_seconds'],1),flush=True)
    return meta


def prepare_layout(record):
    directory = ROOT/'layouts'/record['capture']
    done = directory/'all_layers.json'
    if done.exists():
        assert json.loads(done.read_text())['prompt_hash']==record['prompt_hash']
        return
    directory.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    cached = {16:[],32:[]}
    spans = {cap:np.asarray(record['geometry'][str(cap)]['spans']) for cap in cached}
    for layer in range(64):
        data = torch.load(ROOT/'captures'/record['capture']/f'layer_{layer:02d}.pt',weights_only=True)
        key = data['k'].cuda()
        for cap in cached:
            cached[cap].append(runtime.envelopes(key,spans[cap]))
        del data,key
    for cap in cached:
        lengths = spans[cap][:,1]-spans[cap][:,0]+1
        lo,hi = runtime.middle_chunk_bounds(lengths,cap*2,cap*4,False)
        offsets = np.r_[0,lengths.cumsum()]
        mx = torch.stack([r[0] for r in cached[cap]])
        mn = torch.stack([r[1] for r in cached[cap]])
        torch.save(dict(center=torch.stack([r[2] for r in cached[cap]]).cpu(),
                        width=torch.stack([r[3] for r in cached[cap]]).cpu()),directory/f'cap{cap}_summaries.pt')
        results = [[] for _ in range(64)]
        for anchor,alpha in CONFIGS:
            factor = torch.as_tensor((anchor/lengths[lo:hi])**alpha,dtype=torch.float32,device='cuda')
            first,last,perm,events,stats = bounded_layout(runtime.extension('packing'),runtime.extension('linkage'),
                mx[:,lo:hi],mn[:,lo:hi],factor,lengths,lo,hi,cap*2)
            full = torch.arange(record['prompt_tokens'],device='cuda',dtype=torch.int32)[None,None].repeat(64,8,1)
            full[:,:,offsets[lo]:offsets[hi]] = perm+int(offsets[lo])
            first,last,full = first.cpu(),last.cpu(),full.cpu()
            for layer in range(64):
                results[layer].append((first[layer].clone(),last[layer].clone(),full[layer].clone()))
        for layer in range(64):
            data = dict(first=torch.stack([r[0] for r in results[layer]]),
                        last=torch.stack([r[1] for r in results[layer]]),
                        permutation=torch.stack([r[2] for r in results[layer]]))
            path = directory/f'cap{cap}_layer{layer:02d}.pt'
            temp=path.with_suffix('.tmp');torch.save(data,temp);temp.replace(path)
        del mx,mn,results,first,last,full,perm,data
    del cached
    gc.collect();torch.cuda.empty_cache()
    atomic_json(done,dict(complete=True,prompt_hash=record['prompt_hash'],elapsed_seconds=time.monotonic()-start))
    print('LAYOUT',record['task'],round(time.monotonic()-start,1),flush=True)


def decode_group(model, tokenizer, records, first, max_new_tokens=1024):
    # Preserve production cache shapes even for a short regression run.
    # FlashAttention kernel tiling can change when its allocated cache shrinks.
    capacity=max(1024,max_new_tokens)+8
    children = [ReplayAttention(r,capacity) for r in records]
    from replay_grouped import GroupAttention
    modeling_qwen3.ALL_ATTENTION_FUNCTIONS['sdpa'] = GroupAttention(children)
    graph,logits,inputs,positions = make_graph(model,children,1)
    count = len(records)
    trajectories = [[[token] for _ in VARIANTS] for token in first]
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos,int) else eos
    done = np.array([[token in eos]*len(VARIANTS) for token in first])
    current = torch.tensor(first,device='cuda')[:,None].repeat(1,len(VARIANTS)).reshape(-1,1)
    start = time.monotonic()
    for step in range(max_new_tokens-1):
        for i,child in enumerate(children):
            child.step.fill_(step)
            positions[i*41:(i+1)*41].fill_(child.plen+step)
        inputs.copy_(current)
        graph.replay()
        current = logits.argmax(-1)
        tokens = current.reshape(count,41).cpu().tolist()
        for i in range(count):
            for j,token in enumerate(tokens[i]):
                if not done[i,j]:
                    trajectories[i][j].append(token)
                    done[i,j] = token in eos
        if step%128==0:
            print('DECODE',[r['task'] for r in records],step,done.sum(1).tolist(),round(time.monotonic()-start,1),flush=True)
        if done.all():
            break
    results = []
    for i,record in enumerate(records):
        variants = []
        for j,config in enumerate(VARIANTS):
            text = tokenizer.decode(trajectories[i][j],skip_special_tokens=True)
            variants.append(dict(**config,tokens=trajectories[i][j],text=text,finished=bool(done[i,j]),**action_info(text)))
        results.append(dict(complete=True,task=record['task'],capture=record['capture'],
            prompt_hash=record['prompt_hash'],variants=variants,elapsed_seconds=time.monotonic()-start,
            group_prompts=count,max_new_tokens=max_new_tokens,cache_capacity=capacity))
    del graph,logits,inputs,positions,current,children
    modeling_qwen3.ALL_ATTENTION_FUNCTIONS['sdpa'] = ORIGINAL_ATTENTION
    gc.collect();torch.cuda.empty_cache()
    return results


def regression_control(model,tokenizer):
    dest = ROOT/'control/regression.json'
    if dest.exists():
        assert json.loads(dest.read_text())['passed']
        return
    old_records = json.loads((OLD/'geometry.json').read_text())
    # Reproduce the original batch memberships and row positions. The first
    # mixed-batch control remains archived separately as a sensitivity check.
    groups = [['discord10/req_009','discord10/req_010'],
              ['discord2048/req_006','discord2048/req_007']]
    replay_actions.ROOT = OLD
    checks = []
    for captures in groups:
        records = [next(r for r in old_records if r['capture']==capture) for capture in captures]
        first,reference = [],[]
        for record in records:
            saved = torch.load(OLD/'captures'/record['capture']/'dense_logits.pt',weights_only=True)
            first.append(int(saved[0].argmax()))
            reference.append(json.loads((OLD/'replay'/(record['capture'].replace('/','_')+'.json')).read_text()))
        results = decode_group(model,tokenizer,records,first,max_new_tokens=32)
        for result,old in zip(results,reference):
            for new,expected in zip(result['variants'],old['variants']):
                assert (new['cap'],new['A'],new['alpha'])==(expected['cap'],expected['A'],expected['alpha'])
                n = min(32,len(expected['tokens']))
                checks.append(dict(task=result['task'],capture=result['capture'],cap=new['cap'],A=new['A'],alpha=new['alpha'],
                    compared_tokens=n,equal=new['tokens'][:n]==expected['tokens'][:n],
                    new_tokens=new['tokens'][:n],reference_tokens=expected['tokens'][:n]))
    replay_actions.ROOT = ROOT
    passed = all(r['equal'] for r in checks)
    atomic_json(dest,dict(passed=passed,checks=checks,description='First 32 tokens vs previous complete paired replay, original batch membership and row positions'))
    assert passed, 'Replay regression failed. Do not start new measurement.'
    print('REGRESSION PASSED',len(checks),'branches',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(MODEL)
    records=prepare(tokenizer)
    if args.prepare_only:
        return
    if args.limit:
        records=records[:args.limit]
    remaining=[r for r in records if not (ROOT/'replay'/(r['capture'].replace('/','_')+'.json')).exists()]
    if not remaining:
        print('Already complete',len(records),flush=True)
        return
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    assert model.config.num_hidden_layers==64 and model.config.num_key_value_heads==8
    regression_control(model,tokenizer)
    provenance=dict(script_sha256=sha(__file__),adapter_sha256=sha(OLD/'experiments/replay_actions.py'),
        grouping_sha256=sha(OLD/'experiments/replay_grouped.py'),runtime_sha256=sha(OLD/'experiments/runtime.py'),
        input_manifest_sha256=sha(ROOT/'inputs/candidate_manifest.json'),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        model='Qwen/Qwen3-32B',dtype='bfloat16',budget_ratio=.125,
        generation='Greedy, shared dense prefill; independent all-layer sparse generated states')
    atomic_json(ROOT/'runtime_manifest.json',provenance)
    for start in range(0,len(remaining),2):
        group=remaining[start:start+2]
        first=[]
        for row in group:
            first.append(capture_prefix(model,row)['first_token'])
            prepare_layout(row)
        for result in decode_group(model,tokenizer,group,first):
            result.update(provenance=provenance)
            atomic_json(ROOT/'replay'/(result['capture'].replace('/','_')+'.json'),result)
            print('COMPLETE',result['task'],flush=True)
        completed=list((ROOT/'replay').glob('*.json'))
        atomic_json(ROOT/'progress.json',dict(completed_tasks=len(completed),planned_tasks=64,
                    complete=len(completed)==64,last_group=[r['task'] for r in group]))


if __name__=='__main__':
    with torch.inference_mode():
        main()
