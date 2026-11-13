"""Extend legacy page32 results to 24 settings and 64 distinct tasks."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers.models.qwen3 import modeling_qwen3

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/workspace/calibration32_20261112')
sys.path.insert(0,str(OLD/'experiments'))
spec=importlib.util.spec_from_file_location('legacy_driver',OLD/'experiments/run_replay.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
runtime=previous.runtime
from replay_attention import Page32Attention,GroupAttention,make_graph,gather_extension,reference_gather
from bsa_sglang.packing_gpu import bounded_layout

SETTINGS=json.loads((ROOT/'settings.json').read_text())
CONFIGS=[(a,b) for a in SETTINGS['anchors'] for b in SETTINGS['alphas']]
LEGACY=[(a,b) for a in (1,4,8,16,32) for b in (.125,.25,.375,.5)]
MISSING=[c for c in CONFIGS if c not in LEGACY]
INTERNAL=LEGACY+MISSING
assert len(CONFIGS)==24 and len(MISSING)==9 and len(INTERNAL)==29
assert SETTINGS['page_size']==32 and SETTINGS['maximum_chunk_size']==16
assert SETTINGS['page_budget_ratio']==.125 and SETTINGS['merge_target']==6
atomic_json=previous.atomic_json;sha=previous.sha
ORIGINAL_ATTENTION=previous.ORIGINAL_ATTENTION
SOURCES={name:sha(Path(__file__).parent/name) for name in
 ('run_replay.py','replay_attention.py','fused_kv_gather.cu','layouts.py','prepare_inputs.py')}
layout_spec=importlib.util.spec_from_file_location('page32_layouts',Path(__file__).with_name('layouts.py'))
layouts=importlib.util.module_from_spec(layout_spec);layout_spec.loader.exec_module(layouts)
for name in ('ROOT','torch','np','time','gc','json','runtime','INTERNAL','atomic_json','sha','bounded_layout','Path'):
    setattr(layouts,name,globals()[name])
prepare_layout=layouts.prepare_layout


def validate_copy():
    torch.manual_seed(20261113);ext=gather_extension();checks=[]
    for plen in (257,8550):
        capacity=1032;pages=(plen+capacity+31)//32
        pk=torch.randn((plen,8,128),device='cuda',dtype=torch.bfloat16);pv=torch.randn_like(pk)
        hk=torch.randn((30,8,capacity,128),device='cuda',dtype=torch.bfloat16);hv=torch.randn_like(hk)
        for start,n in ((0,20),(20,9)):
            permutation=(torch.randperm(plen,device='cuda')[None,None]+
                torch.arange(n*8,device='cuda').reshape(n,8,1))%plen
            permutation=permutation.int().contiguous()
            whole=torch.randint(-1,pages,(n,8,pages),device='cuda',dtype=torch.int32)
            local=whole[:,:,:min(pages,39)]
            local[:,:,0]=-1;local[:,:,1]=plen//32;local[:,:,-1]=pages-1
            expected=reference_gather(local,permutation,pk,pv,hk[start:start+n],hv[start:start+n],32)
            actual=ext.gather(local,permutation,pk,pv,hk,hv,start,32)
            assert all(torch.equal(a,b) for a,b in zip(expected,actual))
            checks.append(dict(plen=plen,count=n,start=start,bitwise_equal=True))
        del pk,pv,hk,hv,permutation,whole,local,expected,actual
        gc.collect();torch.cuda.empty_cache()
    atomic_json(ROOT/'control/copy_validation.json',dict(passed=True,checks=checks,sources=SOURCES))
    print('COPY VALIDATION PASSED',flush=True)


def first_token(record):
    tensor=torch.load(Path(record['source_root'])/'captures'/record['capture']/record['first_logits_name'],weights_only=True)
    if record['first_logits_row'] is not None:tensor=tensor[record['first_logits_row']]
    return int(tensor.argmax())


def decode(model,tokenizer,records,max_new_tokens=1024):
    children=[Page32Attention(ROOT,r,INTERNAL,r['cache_capacity'],True) for r in records]
    modeling_qwen3.ALL_ATTENTION_FUNCTIONS['sdpa']=GroupAttention(children)
    graph,logits,inputs,positions=make_graph(model,children)
    first=[first_token(r) for r in records]
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    trajectories=[[[t] for _ in range(41)] for t in first]
    done=np.asarray([[t in eos]*41 for t in first]);done[:,29:40]=True
    active=list(range(29))+[40]
    current=torch.tensor(first,device='cuda')[:,None].repeat(1,41).reshape(-1,1)
    start=time.monotonic()
    for step in range(max_new_tokens-1):
        for i,child in enumerate(children):
            child.step.fill_(step);positions[i*41:(i+1)*41].fill_(child.plen+step)
        inputs.copy_(current);graph.replay();current=logits.argmax(-1)
        tokens=current.reshape(len(records),41).cpu().tolist()
        for i in range(len(records)):
            for j in active:
                if not done[i,j]:
                    token=tokens[i][j];trajectories[i][j].append(token);done[i,j]=token in eos
        if step%128==0:
            print('DECODE',[r['task'] for r in records],step,(done[:,active].sum(1)).tolist(),round(time.monotonic()-start,1),flush=True)
        if done.all():break
    elapsed=time.monotonic()-start;results=[]
    for i,r in enumerate(records):
        variants=[]
        for j,(a,b) in list(enumerate(INTERNAL))+[(40,(0,0))]:
            text=tokenizer.decode(trajectories[i][j],skip_special_tokens=True)
            variants.append(dict(cap=0 if j==40 else 16,A=a,alpha=b,tokens=trajectories[i][j],
                text=text,finished=bool(done[i,j]),**previous.action_info(text)))
        results.append(dict(task=r['task'],capture=r['capture'],variants=variants,elapsed_seconds=elapsed))
    del graph,logits,inputs,positions,current,children
    modeling_qwen3.ALL_ATTENTION_FUNCTIONS['sdpa']=ORIGINAL_ATTENTION
    gc.collect();torch.cuda.empty_cache()
    return results


def compare_legacy(record,result,prefix=None):
    path=Path(record['reuse_path']);assert sha(path)==record['reuse_sha256']
    old=json.loads(path.read_text())
    reference={(v['cap'],v['A'],v['alpha']):v for v in old['variants'] if v['cap'] in (0,16)}
    checks=[]
    for v in result['variants']:
        key=(v['cap'],v['A'],v['alpha'])
        if key not in reference:continue
        expected=reference[key]
        a,b=v['tokens'],expected['tokens']
        if prefix is not None:a,b=a[:prefix],b[:prefix]
        checks.append(dict(cap=v['cap'],A=v['A'],alpha=v['alpha'],equal=a==b,
            compared_tokens=len(b),first_mismatch=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)))
    assert len(checks)==21
    return checks


def validate_model(model,tokenizer,by_capture,groups):
    path=ROOT/'control/model_validation.json'
    if path.exists():
        existing=json.loads(path.read_text());assert existing['passed'] and existing['sources']==SOURCES
        return
    checks=[]
    # Both the old single-prompt cache shape and old two-prompt schedule.
    for index in (0,3,10,11):
        records=[by_capture[c] for c in groups[index]]
        for r in records:prepare_layout(r)
        results=decode(model,tokenizer,records,max_new_tokens=32)
        for r,result in zip(records,results):
            if not r.get('reuse_path'):continue
            checks.extend(dict(task=r['task'],capture=r['capture'],**c) for c in compare_legacy(r,result,prefix=32))
    passed=all(c['equal'] for c in checks)
    atomic_json(path,dict(passed=passed,checks=checks,sources=SOURCES,
        description='Legacy cap16 plus dense token prefixes at their original model row positions'))
    assert passed,'Legacy prefix validation failed. Do not combine these results.'
    print('LEGACY MODEL VALIDATION PASSED',len(checks),flush=True)


def progress(manifest):
    ready={r['capture'] for r in manifest['records'] if r['included'] and
           (ROOT/'replay'/(r['capture'].replace('/','_')+'.json')).exists()}
    complete=[task for task in manifest['tasks'] if all(r['capture'] in ready for r in manifest['records'] if r['included'] and r['task']==task)]
    atomic_json(ROOT/'progress.json',dict(completed_tasks=len(complete),planned_tasks=64,
        completed_steps=len(ready),planned_steps=72,complete=len(complete)==64,
        tasks=complete,prior_tasks=13,prior_steps=21))


def main(model,tokenizer):
    validate_copy()
    manifest=json.loads((ROOT/'inputs/records.json').read_text());by_capture={r['capture']:r for r in manifest['records']}
    validate_model(model,tokenizer,by_capture,manifest['groups'])
    provenance=dict(sources=SOURCES,settings_sha256=sha(ROOT/'settings.json'),
        manifest_sha256=sha(ROOT/'inputs/manifest.json'),torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        reused_grid=LEGACY,reported_grid=CONFIGS,missing_grid=MISSING)
    atomic_json(ROOT/'runtime_manifest.json',provenance);progress(manifest)
    for group in manifest['groups']:
        records=[by_capture[c] for c in group]
        paths=[ROOT/'replay'/(r['capture'].replace('/','_')+'.json') for r in records]
        if all(p.exists() for p in paths):continue
        for r in records:prepare_layout(r)
        results=decode(model,tokenizer,records)
        for record,generated,path in zip(records,results,paths):
            checks=[];reference={}
            if record.get('reuse_path'):
                checks=compare_legacy(record,generated)
                audit=ROOT/'control/full_reuse_checks'/(record['capture'].replace('/','_')+'.json')
                atomic_json(audit,dict(passed=all(c['equal'] for c in checks),checks=checks))
                if not all(c['equal'] for c in checks):
                    atomic_json(ROOT/'control/failed_replay.json',generated)
                    raise RuntimeError('Full legacy replay differed. Refusing to merge settings: '+record['capture'])
                old=json.loads(Path(record['reuse_path']).read_text())
                reference={(v['A'],v['alpha']):v for v in old['variants'] if v['cap']==16}
                dense=old['variants'][-1]
            else:dense=generated['variants'][-1]
            produced={(v['A'],v['alpha']):v for v in generated['variants'] if v['cap']==16}
            variants=[dict(reference.get(c,produced[c]),measurement_source='reused' if c in reference else 'new') for c in CONFIGS]
            variants.append(dict(dense,measurement_source='reused' if record.get('reuse_path') else 'new'))
            output=dict(complete=True,task=record['task'],capture=record['capture'],included=record['included'],
                task_order=record['task_order'],prompt_hash=record['prompt_hash'],variants=variants,
                max_new_tokens=1024,cache_capacity=record['cache_capacity'],elapsed_seconds=generated['elapsed_seconds'],
                group=group,settings=SETTINGS,provenance=provenance,reuse_validation_passed=bool(checks) if checks else None,
                reuse_path=record.get('reuse_path'),reuse_sha256=record.get('reuse_sha256'))
            atomic_json(path,output)
            print('COMPLETE',record['task'],record['capture'],'included',record['included'],flush=True)
        progress(manifest)
    print('ALL 64 TASKS COMPLETE',flush=True)
