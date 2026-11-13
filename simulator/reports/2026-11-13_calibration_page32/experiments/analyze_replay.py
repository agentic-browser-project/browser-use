"""Aggregate complete grids by task and preserve every rolling search."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from subset_search import optimize,TARGET,SECONDARY

ROOT=Path(__file__).resolve().parents[1]
SETTINGS=json.loads((ROOT/'settings.json').read_text())
CONFIGS=[(a,b) for a in SETTINGS['anchors'] for b in SETTINGS['alphas']]
METRICS=('target_equal','type_equal','action_equal','finished','parsed')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def rank(frame,metric):
    result=frame.groupby(['A','alpha'])[metric].agg(['mean','count']).reset_index()
    result['rank']=result['mean'].round(12).rank(method='min',ascending=False).astype('Int64')
    return result.sort_values(['rank','A','alpha'])


def main():
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    included=[r for r in manifest['records'] if r['included']]
    paths={r['capture']:ROOT/'replay'/(r['capture'].replace('/','_')+'.json') for r in included}
    complete=[t for t in manifest['tasks'] if all(paths[r['capture']].exists() for r in included if r['task']==t)]
    assert len(complete)>=16
    rows=[];metadata=[];hashes={}
    for record in included:
        if record['task'] not in complete:continue
        path=paths[record['capture']];raw=path.read_bytes();hashes[path.name]=hashlib.sha256(raw).hexdigest();r=json.loads(raw)
        assert r['complete'] and r['included'] and r['prompt_hash']==record['prompt_hash']
        assert r['settings']==SETTINGS and r['max_new_tokens']==1024
        assert [(v['cap'],v['A'],v['alpha']) for v in r['variants']]==[(16,a,b) for a,b in CONFIGS]+[(0,0,0)]
        dense=r['variants'][-1];valid_dense=bool(dense['finished'] and dense['parsed'])
        indexed=bool(valid_dense and dense['indices'])
        metadata.append(dict(**record,dense_valid=valid_dense,dense_index_eligible=indexed,
            dense_actions=dense['actions'],dense_types=dense['types'],dense_indices=dense['indices']))
        for v in r['variants'][:-1]:
            valid=bool(v['finished'] and v['parsed']);type_match=valid and v['types']==dense['types']
            joint=type_match and v['indices']==dense['indices']
            rows.append(dict(task=r['task'],capture=r['capture'],task_order=record['task_order'],A=v['A'],alpha=v['alpha'],
                target_equal=float(joint) if indexed else np.nan,type_equal=float(type_match) if valid_dense else np.nan,
                action_equal=float(valid and v['actions']==dense['actions']) if valid_dense else np.nan,
                finished=float(v['finished']),parsed=float(valid),measurement_source=v['measurement_source']))
    states=pd.DataFrame(rows)
    tasks=states.groupby(['task','task_order','A','alpha'])[list(METRICS)].mean().reset_index()
    counts=states.groupby(['task','A','alpha']).agg(included_steps=('capture','size'),
        joint_eligible_steps=('target_equal','count'),type_eligible_steps=('type_equal','count')).reset_index()
    tasks=tasks.merge(counts,on=['task','A','alpha'],validate='one_to_one')
    output=ROOT/'results';output.mkdir(exist_ok=True)
    states.to_csv(output/'per_step.csv',index=False);tasks.to_csv(output/'per_task.csv',index=False)
    save(output/'step_metadata.json',metadata)
    matrix=tasks.pivot(index='task',columns=['A','alpha'],values='target_equal').dropna(how='all')
    matrix=matrix.reindex(columns=pd.MultiIndex.from_tuples(CONFIGS));assert not matrix.isna().any().any()
    groups={'measured_pool':set(complete)}
    for size in (16,32):
        prefix=set(manifest['tasks'][:size])
        if prefix.issubset(complete):groups[f'primary{size}']=prefix
    searches={};membership=[]
    for size in SETTINGS['subset_sizes']:
        if len(matrix)<size:
            searches[str(size)]=dict(outcome='Insufficient eligible tasks',eligible_pool_tasks=len(matrix),subset_tasks=size)
            continue
        search=optimize(matrix,size);searches[str(size)]=search
        if 'task_ids' not in search:continue
        selected=set(search['task_ids']);groups[f'exploratory{size}']=selected
        groups[f'complement{size}']=set(complete)-selected
        membership.extend(dict(task=t,subset_size=size,selected=t in selected) for t in complete)
    ranks=[]
    for cohort,ids in groups.items():
        for metric in METRICS:
            result=rank(tasks[tasks.task.isin(ids)],metric);result['cohort']=cohort;result['metric']=metric
            ranks.append(result)
    pd.concat(ranks,ignore_index=True).to_csv(output/'rankings.csv',index=False)
    pd.DataFrame(membership,columns=['task','subset_size','selected']).to_csv(output/'subset_membership.csv',index=False)
    save(output/'subset_search.json',searches)
    status=dict(complete=len(complete)==64,completed_tasks=len(complete),completed_steps=len(metadata),
        eligible_tasks=len(matrix),planned_tasks=64,planned_steps=72,settings=SETTINGS,
        task_ids=complete,replay_hashes=hashes,analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        optimizer_sha256=hashlib.sha256(Path(__file__).with_name('subset_search.py').read_bytes()).hexdigest())
    save(output/'analysis_manifest.json',status)
    checkpoint=output/'checkpoints'/f'n{len(complete):03d}';checkpoint.mkdir(parents=True,exist_ok=True)
    for name in ('analysis_manifest.json','rankings.csv','subset_search.json','subset_membership.csv'):
        shutil.copyfile(output/name,checkpoint/name)
    print(json.dumps(dict(completed_tasks=len(complete),completed_steps=len(metadata),eligible_tasks=len(matrix),
        searches={s:{k:v for k,v in r.items() if k not in ('task_ids','objective_order')} for s,r in searches.items()})),flush=True)
    from write_report import main as write_report
    write_report()


if __name__=='__main__':main()
