"""Fixed 64-step WebVoyager replay, 24 settings, page64/max32 only."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/workspace/calibration32_20261112')
sys.path.insert(0,str(OLD/'experiments'))
# Import under an explicit name to avoid importing this new run_replay module.
import importlib.util
spec=importlib.util.spec_from_file_location('previous_driver',OLD/'experiments/run_replay.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
previous.ROOT=ROOT
runtime=previous.runtime;runtime.ROOT=ROOT
from page64_attention import Page64Attention,GroupAttention,make_graph,gather_extension,reference_gather
from bsa_sglang.packing_gpu import bounded_layout

SETTINGS=json.loads((ROOT/'settings.json').read_text())
CONFIGS=[(a,b) for a in SETTINGS['anchors'] for b in SETTINGS['alphas']]
VARIANTS=[dict(cap=32,A=a,alpha=b) for a,b in CONFIGS]+[dict(cap=0,A=0,alpha=0)]
BATCH=len(VARIANTS)
assert len(CONFIGS)==24 and SETTINGS['page_size']==64 and SETTINGS['maximum_chunk_size']==32
assert SETTINGS['page_budget_ratio']==.125 and SETTINGS['merge_target']==6
assert SETTINGS['tasks_per_batch']==2 and SETTINGS['max_new_tokens']==1024
atomic_json=previous.atomic_json
sha=previous.sha
ORIGINAL_ATTENTION=previous.ORIGINAL_ATTENTION
SOURCES={name:sha(Path(__file__).parent/name) for name in
         ('run_replay.py','page64_attention.py','fused_kv_gather.cu')}


def validate_copy():
    destination=ROOT/'control/copy_validation.json'
    extension=gather_extension();checks=[]
    torch.manual_seed(20261112)
    for plen in (257,8550):
        capacity=1032;pages=(plen+capacity+63)//64
        pk=torch.randn((plen,8,128),device='cuda',dtype=torch.bfloat16);pv=torch.randn_like(pk)
        hk=torch.randn((BATCH,8,capacity,128),device='cuda',dtype=torch.bfloat16);hv=torch.randn_like(hk)
        permutation=(torch.randperm(plen,device='cuda')[None,None]+
                     torch.arange(len(CONFIGS)*8,device='cuda').reshape(len(CONFIGS),8,1))%plen
        permutation=permutation.int().contiguous()
        whole=torch.randint(-1,pages,(len(CONFIGS),8,pages),device='cuda',dtype=torch.int32)
        local=whole[:,:,:min(pages,39)]
        local[:,:,0]=-1;local[:,:,1]=plen//64;local[:,:,-1]=pages-1
        expected=reference_gather(local,permutation,pk,pv,hk,hv,64)
        actual=extension.gather(local,permutation,pk,pv,hk,hv,0,64)
        assert all(torch.equal(a,b) for a,b in zip(actual,expected))
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured=extension.gather(local,permutation,pk,pv,hk,hv,0,64)
        graph.replay();torch.cuda.synchronize()
        assert all(torch.equal(a,b) for a,b in zip(captured,expected))
        checks.append(dict(prompt_tokens=plen,configurations=len(CONFIGS),bitwise_equal=True,
                           graph_bitwise_equal=True,noncontiguous_pages=not local.is_contiguous()))
        del graph,captured,actual,expected,pk,pv,hk,hv,permutation,whole,local
        gc.collect();torch.cuda.empty_cache()
    atomic_json(destination,dict(passed=True,checks=checks,cuda_source_sha256=SOURCES['fused_kv_gather.cu']))
    print('COPY VALIDATION PASSED',flush=True)


def prepare_layout(record):
    directory=ROOT/'layouts'/record['capture'];done=directory/'all_layers.json'
    if done.exists():
        saved=json.loads(done.read_text())
        assert saved['prompt_hash']==record['prompt_hash'] and saved['settings_sha256']==sha(ROOT/'settings.json')
        return
    directory.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    spans=np.asarray(record['geometry']['32']['spans']);lengths=spans[:,1]-spans[:,0]+1
    offsets=np.r_[0,lengths.cumsum()]
    lo,hi=runtime.middle_chunk_bounds(lengths,64,128,False)
    summaries=[]
    for layer in range(64):
        data=torch.load(ROOT/'captures'/record['capture']/f'layer_{layer:02d}.pt',weights_only=True)
        keys=data['k'].cuda();summaries.append(runtime.envelopes(keys,spans));del data,keys
    mx=torch.stack([r[0] for r in summaries]);mn=torch.stack([r[1] for r in summaries])
    torch.save(dict(center=torch.stack([r[2] for r in summaries]).cpu(),
                    width=torch.stack([r[3] for r in summaries]).cpu()),directory/'cap32_summaries.pt')
    results=[[] for _ in range(64)]
    for anchor,alpha in CONFIGS:
        factor=torch.as_tensor((anchor/lengths[lo:hi])**alpha,device='cuda',dtype=torch.float32)
        first,last,perm,events,stats=bounded_layout(runtime.extension('packing'),runtime.extension('linkage'),
            mx[:,lo:hi],mn[:,lo:hi],factor,lengths,lo,hi,64)
        full=torch.arange(record['prompt_tokens'],device='cuda',dtype=torch.int32)[None,None].repeat(64,8,1)
        full[:,:,offsets[lo]:offsets[hi]]=perm+int(offsets[lo])
        first,last,full=first.cpu(),last.cpu(),full.cpu()
        for layer in range(64):
            results[layer].append((first[layer].clone(),last[layer].clone(),full[layer].clone()))
    for layer in range(64):
        data=dict(first=torch.stack([r[0] for r in results[layer]]),
                  last=torch.stack([r[1] for r in results[layer]]),
                  permutation=torch.stack([r[2] for r in results[layer]]))
        path=directory/f'cap32_layer{layer:02d}.pt';temp=path.with_suffix('.tmp')
        torch.save(data,temp);temp.replace(path)
    atomic_json(done,dict(complete=True,prompt_hash=record['prompt_hash'],settings_sha256=sha(ROOT/'settings.json'),
        page_size=64,maximum_chunk_size=32,configurations=CONFIGS,elapsed_seconds=time.monotonic()-start))
    print('LAYOUT',record['task'],round(time.monotonic()-start,1),flush=True)
    del summaries,mx,mn,results,data,first,last,full,perm
    gc.collect();torch.cuda.empty_cache()


def decode(model,tokenizer,records,first,optimized=True,max_new_tokens=1024):
    children=[Page64Attention(ROOT,r,CONFIGS,1032,optimized) for r in records]
    modeling_qwen3.ALL_ATTENTION_FUNCTIONS['sdpa']=GroupAttention(children)
    graph,logits,inputs,positions=make_graph(model,children)
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    trajectories=[[[t] for _ in VARIANTS] for t in first]
    done=np.asarray([[t in eos]*BATCH for t in first])
    current=torch.tensor(first,device='cuda')[:,None].repeat(1,BATCH).reshape(-1,1)
    start=time.monotonic()
    for step in range(max_new_tokens-1):
        for i,child in enumerate(children):
            child.step.fill_(step);positions[i*BATCH:(i+1)*BATCH].fill_(child.plen+step)
        inputs.copy_(current);graph.replay();current=logits.argmax(-1)
        tokens=current.reshape(len(records),BATCH).cpu().tolist()
        for i in range(len(records)):
            for j,t in enumerate(tokens[i]):
                if not done[i,j]:
                    trajectories[i][j].append(t);done[i,j]=t in eos
        if step%128==0:
            print('DECODE',[r['task'] for r in records],step,done.sum(1).tolist(),round(time.monotonic()-start,1),flush=True)
        if done.all():break
    elapsed=time.monotonic()-start;results=[]
    for i,record in enumerate(records):
        variants=[]
        for j,config in enumerate(VARIANTS):
            text=tokenizer.decode(trajectories[i][j],skip_special_tokens=True)
            variants.append(dict(**config,tokens=trajectories[i][j],text=text,finished=bool(done[i,j]),
                                 **previous.action_info(text)))
        results.append(dict(complete=True,task=record['task'],capture=record['capture'],
            prompt_hash=record['prompt_hash'],variants=variants,elapsed_seconds=elapsed,
            max_new_tokens=max_new_tokens,cache_capacity=1032,group_tasks=[r['task'] for r in records],
            optimized=optimized,settings=SETTINGS,settings_sha256=sha(ROOT/'settings.json')))
    del graph,logits,inputs,positions,current,children
    modeling_qwen3.ALL_ATTENTION_FUNCTIONS['sdpa']=ORIGINAL_ATTENTION
    gc.collect();torch.cuda.empty_cache()
    return results


def validate_model(model,tokenizer,records):
    path=ROOT/'control/model_validation.json'
    if path.exists():
        saved=json.loads(path.read_text())
        assert saved['passed'] and saved['sources']==SOURCES and saved['settings_sha256']==sha(ROOT/'settings.json')
        return
    checks=[];timings=[]
    for begin in (0,2):
        group=records[begin:begin+2];first=[]
        for record in group:
            first.append(previous.capture_prefix(model,record)['first_token']);prepare_layout(record)
        reference=decode(model,tokenizer,group,first,optimized=False,max_new_tokens=32)
        optimized=decode(model,tokenizer,group,first,optimized=True,max_new_tokens=32)
        for a,b in zip(reference,optimized):
            for va,vb in zip(a['variants'],b['variants']):
                checks.append(dict(task=a['task'],A=va['A'],alpha=va['alpha'],cap=va['cap'],
                    equal=va['tokens']==vb['tokens'],reference_tokens=va['tokens'],optimized_tokens=vb['tokens']))
        timings.append(dict(tasks=[r['task'] for r in group],reference_seconds=reference[0]['elapsed_seconds'],
            optimized_seconds=optimized[0]['elapsed_seconds'],generated_steps=31,branches=2*BATCH))
    passed=all(r['equal'] for r in checks)
    atomic_json(path,dict(passed=passed,checks=checks,timings=timings,sources=SOURCES,
        settings_sha256=sha(ROOT/'settings.json'),scope='Matched batch first 32 generated tokens, not exhaustive long-response equality'))
    assert passed,'Optimized full-model validation failed. Do not begin measurement.'
    print('MODEL VALIDATION PASSED',len(checks),json.dumps(timings),flush=True)


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    validate_copy()
    tokenizer=AutoTokenizer.from_pretrained('/models/Qwen3-32B');records=previous.prepare(tokenizer)
    model=AutoModelForCausalLM.from_pretrained('/models/Qwen3-32B',dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    assert model.config.num_hidden_layers==64 and model.config.num_key_value_heads==8
    validate_model(model,tokenizer,records)
    provenance=dict(sources=SOURCES,settings_sha256=sha(ROOT/'settings.json'),
        input_manifest_sha256=sha(ROOT/'inputs/candidate_manifest.json'),
        previous_driver_sha256=sha(OLD/'experiments/run_replay.py'),
        runtime_sha256=sha(previous.OLD/'experiments/runtime.py'),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),model=SETTINGS['model'],dtype=SETTINGS['dtype'])
    atomic_json(ROOT/'runtime_manifest.json',provenance)
    for begin in range(0,len(records),2):
        group=records[begin:begin+2]
        paths=[ROOT/'replay'/(r['capture'].replace('/','_')+'.json') for r in group]
        if all(p.exists() for p in paths):
            for p in paths:assert json.loads(p.read_text())['settings_sha256']==sha(ROOT/'settings.json')
            continue
        if any(p.exists() for p in paths):
            for p in paths:
                if p.exists():
                    archive=ROOT/'orphan_checkpoints';archive.mkdir(exist_ok=True)
                    p.rename(archive/(p.stem+'_'+sha(p)[:12]+'.json'))
        first=[]
        for record in group:
            first.append(previous.capture_prefix(model,record)['first_token']);prepare_layout(record)
        results=decode(model,tokenizer,group,first)
        for path,result in zip(paths,results):
            result['provenance']=provenance;atomic_json(path,result)
            print('COMPLETE',result['task'],flush=True)
        count=len(list((ROOT/'replay').glob('*.json')))
        atomic_json(ROOT/'progress.json',dict(completed_tasks=count,planned_tasks=64,complete=count==64,
                    last_group=[r['task'] for r in group],settings_sha256=sha(ROOT/'settings.json')))
    print('ALL 64 TASKS COMPLETE',flush=True)


if __name__=='__main__':
    with torch.inference_mode():main()
