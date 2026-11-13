"""Run the identical replay adapter on two prompts per weight read on spark00."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3
from replay_actions import ReplayAttention,action_info
from runtime import ROOT,PREVIOUS,CONFIGS


class GroupAttention:
    def __init__(self,children):self.children=children
    def __call__(self,module,query,key,value,mask,**kwargs):
        values=[]
        for i,child in enumerate(self.children):
            sl=slice(i*41,(i+1)*41)
            values.append(child(module,query[sl],key[sl],value[sl],mask,**kwargs)[0])
        return torch.cat(values),None


def make_graph(model,children,width):
    for child in children:
        child.configure(width);child.step.zero_()
    inputs=torch.full((41*len(children),width),151645,dtype=torch.long,device='cuda')
    positions=torch.cat([(child.plen+child.offset[None]).repeat(41,1) for child in children])
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            result=model(input_ids=inputs,position_ids=positions,attention_mask={'full_attention':None},use_cache=False).logits
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result=model(input_ids=inputs,position_ids=positions,attention_mask={'full_attention':None},use_cache=False).logits
    torch.cuda.synchronize()
    return graph,result,inputs,positions


def process_group(model,tokenizer,records,args):
    start=time.monotonic()
    generations=[json.loads((PREVIOUS/'qwen32'/r['capture']/'generation.json').read_text()) for r in records]
    outputs=[g['output_ids'] for g in generations]
    capacity=max(args.max_new_tokens,max(map(len,outputs)))+args.teacher_width
    children=[ReplayAttention(r,capacity) for r in records]
    modeling_qwen3.ALL_ATTENTION_FUNCTIONS['sdpa']=GroupAttention(children)
    count=len(children)
    destinations=[ROOT/'replay'/(r['capture'].replace('/','_')+'.json') for r in records]
    teacher_paths=[p.with_suffix('.teacher.json') for p in destinations]
    teachers=[json.loads(p.read_text()) if p.exists() else [] for p in teacher_paths]
    missing=[not bool(t) for t in teachers]
    saved=[torch.load(ROOT/'captures'/r['capture']/'dense_logits.pt',weights_only=True) for r in records]
    first=[int(s[0].argmax()) for s in saved]
    print('Group ready',[r['capture'] for r in records],'GiB',torch.cuda.memory_allocated()/2**30,flush=True)
    if any(missing):
        graph,logits,inputs,positions=make_graph(model,children,args.teacher_width)
        for offset in range(0,max(len(x)-1 for x in outputs),args.teacher_width):
            for i,child in enumerate(children):
                n=max(0,min(args.teacher_width,len(outputs[i])-1-offset))
                tokens=outputs[i][offset:offset+n]+[151645]*(args.teacher_width-n)
                child.step.fill_(offset)
                positions[i*41:(i+1)*41].copy_((child.plen+offset+child.offset)[None].expand(41,-1))
                inputs[i*41:(i+1)*41].copy_(torch.tensor(tokens,device='cuda')[None].expand(41,-1))
            graph.replay()
            for i,record in enumerate(records):
                n=max(0,min(args.teacher_width,len(outputs[i])-1-offset))
                if n==0 or not missing[i]:continue
                lp=logits[i*41:(i+1)*41,:n].float().log_softmax(-1)
                dense=lp[40]
                kl=(dense.exp()[None]*(dense[None]-lp)).sum(-1)
                labels=torch.tensor(outputs[i][offset+1:offset+n+1],device='cuda')
                nll=-lp.gather(-1,labels[None,:,None].expand(41,-1,1)).squeeze(-1)
                agreement=lp.argmax(-1)==lp[40].argmax(-1)[None]
                gold=saved[i][offset+1:offset+n+1].cuda().float().log_softmax(-1)
                validation=(gold.exp()*(gold-dense)).sum(-1)
                kl,nll,agreement,validation=kl.cpu(),nll.cpu(),agreement.cpu(),validation.cpu()
                for j in range(n):
                    call=offset+j
                    teachers[i].append(dict(call=call,action=call in record['action_calls'],
                        kl=kl[:,j].tolist(),nll=nll[:,j].tolist(),greedy_agreement=agreement[:,j].tolist(),
                        dense_reference_kl=float(validation[j])))
            if offset%128==0:print('Group teacher',records[0]['capture'],offset,'seconds',round(time.monotonic()-start,1),flush=True)
        for path,teacher,out in zip(teacher_paths,teachers,outputs):
            assert len(teacher)==len(out)-1
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(teacher)+'\n');temp.replace(path)
        del graph,logits,inputs,positions,lp,dense,gold
        gc.collect();torch.cuda.empty_cache()
    graph,logits,inputs,positions=make_graph(model,children,1)
    trajectories=[[[token] for _ in range(41)] for token in first]
    done=np.zeros((count,41),dtype=bool)
    current=torch.tensor(first,device='cuda')[:,None].repeat(1,41).reshape(count*41,1)
    eos=model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else eos
    for step in range(args.max_new_tokens-1):
        for i,child in enumerate(children):
            child.step.fill_(step)
            positions[i*41:(i+1)*41].fill_(child.plen+step)
        inputs.copy_(current)
        graph.replay()
        current=logits.argmax(-1)
        ids=current.reshape(count,41).cpu().tolist()
        for i in range(count):
            for j,token in enumerate(ids[i]):
                if not done[i,j]:
                    trajectories[i][j].append(token);done[i,j]=token in eos
        if step%128==0:
            print('Group free',records[0]['capture'],step,'done',done.sum(1).tolist(),
                  'seconds',round(time.monotonic()-start,1),flush=True)
        if done.all():break
    for i,record in enumerate(records):
        variants=[dict(cap=cap,A=a,alpha=b) for cap in (16,32) for a,b in CONFIGS]+[dict(cap=0,A=0,alpha=0)]
        for j,item in enumerate(variants):
            text=tokenizer.decode(trajectories[i][j],skip_special_tokens=True)
            item.update(tokens=trajectories[i][j],text=text,finished=bool(done[i,j]),**action_info(text))
        result=dict(complete=True,capture=record['capture'],task=record['task'],prompt_hash=record['prompt_hash'],
            variants=variants,teacher=teachers[i],historical_action=action_info(tokenizer.decode(outputs[i],skip_special_tokens=True)),
            elapsed_seconds=time.monotonic()-start,group_prompts=count,teacher_width=args.teacher_width,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            adapter_sha256=hashlib.sha256((Path(__file__).parent/'replay_actions.py').read_bytes()).hexdigest(),
            torch=torch.__version__,max_new_tokens=args.max_new_tokens,
            decoder='BF16 FlashAttention; canonical BrowSA mean-query and distinct-page CUDA selectors',
            generation='greedy, shared dense prefix; independent sparse generated states')
        temp=destinations[i].with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False)+'\n');temp.replace(destinations[i])
        print('COMPLETE',record['capture'],round(time.monotonic()-start,1),flush=True)
    modeling_qwen3.ALL_ATTENTION_FUNCTIONS['sdpa']=None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--teacher-width',type=int,default=8)
    parser.add_argument('--max-new-tokens',type=int,default=1024)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    records=json.loads((ROOT/'geometry.json').read_text())
    records=list({r['prompt_hash']:r for r in reversed(records)}.values())[::-1]
    records=[r for r in records if not (ROOT/'replay'/(r['capture'].replace('/','_')+'.json')).exists()]
    groups=[]
    for record in records:
        if record['output_tokens']>1024:
            groups.append([record])
        elif groups and len(groups[-1])==1 and groups[-1][0]['output_tokens']<=1024:
            groups[-1].append(record)
        else:groups.append([record])
    start=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained('/models/Qwen3-32B')
    model=AutoModelForCausalLM.from_pretrained('/models/Qwen3-32B',dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    print('Loaded grouped replay',time.monotonic()-start,'groups',len(groups),flush=True)
    for group in groups:
        process_group(model,tokenizer,group,args)
        gc.collect();torch.cuda.empty_cache()
    (ROOT/'replay/manifest.json').write_text(json.dumps(dict(complete=True,unique_prompts=21,
        remaining_prompts_this_run=len(records),groups_this_run=len(groups),
        elapsed_seconds=time.monotonic()-start),indent=2)+'\n')
    print('ALL COMPLETE',time.monotonic()-start,flush=True)


if __name__=='__main__':
    with torch.inference_mode():main()
