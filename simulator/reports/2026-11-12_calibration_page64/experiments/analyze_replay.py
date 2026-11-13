"""Preserve all outcomes and search declared 16/32-task subsets transparently."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds,LinearConstraint,milp
from action_metrics import METRIC_VERSION,target_indices_equal

ROOT=Path(__file__).resolve().parents[1]
SETTINGS=json.loads((ROOT/'settings.json').read_text())
CONFIGS=[(a,b) for a in SETTINGS['anchors'] for b in SETTINGS['alphas']]
TARGET=(32,.25)
METRICS=('target_equal','target_equal_raw','type_equal','action_equal','finished','parsed')


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def exact_subset(matrix,size):
    """Exact worst-rival match margin, with target match count as tie breaker."""
    assert len(matrix)>=size and not matrix.isna().any().any()
    target=matrix[TARGET].to_numpy()
    difference=target[:,None]-matrix.drop(columns=[TARGET]).to_numpy()
    n,rivals=difference.shape
    constraint=np.vstack([np.column_stack([difference.T,-np.ones(rivals)]),np.r_[np.ones(n),0]])
    result=milp(np.r_[-target,-float(size+1)],integrality=np.ones(n+1),
        bounds=Bounds(np.r_[np.zeros(n),-size],np.r_[np.ones(n),size]),
        constraints=LinearConstraint(constraint,np.r_[np.zeros(rivals),size],np.r_[np.full(rivals,np.inf),size]),
        options={'time_limit':300,'mip_rel_gap':0})
    if not result.success or result.status!=0:
        return dict(outcome='Exact optimization did not finish; no optimum claimed',
                    solver_status=int(result.status),message=result.message,eligible_pool_tasks=n,subset_tasks=size)
    selected=result.x[:-1]>.5
    assert selected.sum()==size and np.allclose(result.x,np.round(result.x),atol=1e-6)
    margin=int(np.round(difference[selected].sum(0).min()))
    assert abs(margin-result.x[-1])<1e-6
    return dict(optimal=True,task_ids=list(matrix.index[selected]),
        minimum_margin_matches=margin,target_matches=int(target[selected].sum()),
        subset_tasks=size,eligible_pool_tasks=n,mip_gap=float(result.mip_gap),
        scipy_version=scipy.__version__,solver='scipy.optimize.milp / HiGHS',
        outcome='unique first' if margin>0 else 'tied first' if margin==0 else
        f'Cannot rank first in any {size}-task subset of this eligible pool')


def random_reference(matrix,size):
    rng=np.random.default_rng(20261112);values=matrix.to_numpy();target=list(matrix.columns).index(TARGET)
    margins=[]
    for _ in range(10000):
        counts=values[rng.choice(len(values),size=size,replace=False)].sum(0)
        margins.append(counts[target]-np.delete(counts,target).max())
    margins=np.asarray(margins)
    return dict(draws=10000,unique_first_fraction=float((margins>0).mean()),
        first_including_ties_fraction=float((margins>=0).mean()),
        interpretation='Uniform subsets within this observed pool, not a probability of generalizing')


def ranking(frame,metric):
    result=frame.groupby(['A','alpha'])[metric].agg(['mean','count']).reset_index()
    result['rank']=result['mean'].round(12).rank(method='min',ascending=False).astype('Int64')
    return result.sort_values(['rank','A','alpha'])


def load_results():
    manifest=json.loads((ROOT/'inputs/candidate_manifest.json').read_text())
    geometry={r['task']:r for r in json.loads((ROOT/'geometry.json').read_text())}
    settings_hash=hashlib.sha256((ROOT/'settings.json').read_bytes()).hexdigest()
    manifest_hash=hashlib.sha256((ROOT/'inputs/candidate_manifest.json').read_bytes()).hexdigest()
    rows=[];metadata=[];actions=[];hashes={}
    for order,info in enumerate(manifest['records']):
        path=ROOT/'replay'/f"webvoyager_new_req_{info['request']:03d}.json"
        if not path.exists():continue
        raw=path.read_bytes();r=json.loads(raw);hashes[path.name]=hashlib.sha256(raw).hexdigest()
        assert r['complete'] and r['task']==info['task'] and r['max_new_tokens']==1024
        assert r['settings_sha256']==settings_hash and r['provenance']['input_manifest_sha256']==manifest_hash
        assert r['prompt_hash']==geometry[info['task']]['prompt_hash']
        variants=r['variants']
        assert [(v['cap'],v['A'],v['alpha']) for v in variants]==[(32,a,b) for a,b in CONFIGS]+[(0,0,0)]
        assert all(0<len(v['tokens'])<=1024 for v in variants)
        dense=variants[-1];dense_valid=bool(dense['finished'] and dense['parsed'])
        eligible=bool(dense_valid and dense['indices'])
        metadata.append(dict(**info,order=order,dense_valid=dense_valid,dense_index_eligible=eligible,
            dense_actions=dense['actions'],dense_types=dense['types'],dense_indices=dense['indices'],
            prompt_tokens=geometry[info['task']]['prompt_tokens'],prompt_hash=r['prompt_hash']))
        for v in variants[:-1]:
            valid=bool(v['finished'] and v['parsed'])
            type_match=valid and v['types']==dense['types']
            joint_match=type_match and target_indices_equal(v['indices'],dense['indices'])
            joint_raw=type_match and v['indices']==dense['indices']
            full_match=valid and v['actions']==dense['actions']
            category=('dense_not_index_eligible' if not eligible else
                'invalid_or_unfinished_sparse' if not valid else 'different_action_types' if not type_match else
                'different_target_indices' if not joint_match else
                'same_types_and_indices_different_arguments' if not full_match else 'complete_action_match')
            rows.append(dict(task=info['task'],website=info['website'],order=order,A=v['A'],alpha=v['alpha'],
                target_equal=float(joint_match) if eligible else np.nan,
                target_equal_raw=float(joint_raw) if eligible else np.nan,
                type_equal=float(type_match) if dense_valid else np.nan,
                action_equal=float(full_match) if dense_valid else np.nan,
                finished=float(v['finished']),parsed=float(valid),generated_tokens=len(v['tokens']),
                agreement_category=category))
            actions.append(dict(task=info['task'],A=v['A'],alpha=v['alpha'],
                                finished=v['finished'],actions=v['actions'],dense_actions=dense['actions']))
    return pd.DataFrame(rows),metadata,actions,hashes


def main():
    data,metadata,actions,hashes=load_results()
    assert len(metadata)>=16,'Rolling search starts after 16 completed tasks'
    output=ROOT/'results';output.mkdir(exist_ok=True)
    data.to_csv(output/'per_task.csv',index=False)
    save_json(output/'task_metadata.json',metadata);save_json(output/'action_comparisons.json',actions)
    groups={'measured_pool':set(data.task),'primary16':set(data[data.order<16].task)}
    if len(metadata)>=32:groups['primary32']=set(data[data.order<32].task)
    matrix=data.pivot(index='task',columns=['A','alpha'],values='target_equal').dropna(how='all')
    matrix=matrix.reindex(columns=pd.MultiIndex.from_tuples(CONFIGS))
    assert not matrix.isna().any().any()
    searches={};membership=[]
    for size in SETTINGS['subset_sizes']:
        if len(matrix)<size:
            searches[str(size)]=dict(outcome='Insufficient dense-index-eligible tasks',
                eligible_pool_tasks=len(matrix),subset_tasks=size)
            continue
        search=exact_subset(matrix,size);searches[str(size)]=search
        if 'task_ids' not in search:continue
        search['random_subset_reference']=random_reference(matrix,size)
        cohort=f'exploratory{size}';selected=set(search['task_ids'])
        groups[cohort]=selected;groups[f'complement{size}']=groups['measured_pool']-selected
        for r in metadata:
            membership.append(dict(task=r['task'],website=r['website'],subset_size=size,selected=r['task'] in selected))
    ranks=[]
    for cohort,tasks in groups.items():
        frame=data[data.task.isin(tasks)]
        for metric in METRICS:
            result=ranking(frame,metric);result['cohort']=cohort;result['metric']=metric
            result['attempted_tasks']=frame.task.nunique();ranks.append(result)
    pd.concat(ranks,ignore_index=True).to_csv(output/'rankings.csv',index=False)
    pd.DataFrame(membership,columns=['task','website','subset_size','selected']).to_csv(output/'subset_membership.csv',index=False)
    save_json(output/'subset_search.json',searches)
    manifest=dict(complete=len(metadata)==64,completed_tasks=len(metadata),planned_tasks=64,
        metric_version=METRIC_VERSION,metric_sha256=hashlib.sha256(Path(__file__).with_name('action_metrics.py').read_bytes()).hexdigest(),
        dense_valid_tasks=sum(r['dense_valid'] for r in metadata),dense_index_eligible_tasks=len(matrix),
        settings=SETTINGS,settings_sha256=hashlib.sha256((ROOT/'settings.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),replay_hashes=hashes)
    save_json(output/'analysis_manifest.json',manifest)
    checkpoint=output/'checkpoints'/f'n{len(metadata):03d}';checkpoint.mkdir(parents=True,exist_ok=True)
    for name in ('analysis_manifest.json','subset_search.json','rankings.csv','subset_membership.csv'):
        shutil.copyfile(output/name,checkpoint/name)
    print(json.dumps(dict(completed_tasks=len(metadata),eligible_tasks=len(matrix),
        searches={k:{x:y for x,y in v.items() if x not in ('task_ids','random_subset_reference')} for k,v in searches.items()})),flush=True)
    from write_report import main as write_report
    write_report()


if __name__=='__main__':main()
