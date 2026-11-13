"""Freeze 64 distinct tasks, preserving all previously measured old-profile states."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HIST=Path('/workspace/semantic_ranking_grid_20261111')
RECENT=Path('/workspace/calibration32_20261112')


def main():
    historical=json.loads((HIST/'geometry.json').read_text())
    historical=list({r['prompt_hash']:r for r in reversed(historical)}.values())[::-1]
    original={r['capture']:r for r in historical}
    historical_groups=[['discord2048/req_001'],
        ['discord2048/req_002','discord2048/req_003'],['discord2048/req_004','discord2048/req_005'],
        ['discord2048/req_006','discord2048/req_007'],['discord2048/req_008','discord2048/req_011'],
        ['discord2048/req_012','discord2048/req_013'],['discord10/req_001','discord10/req_002'],
        ['discord10/req_003','discord10/req_004'],['discord10/req_005','discord2048/req_009'],
        ['discord10/req_007','discord2048/req_010'],['discord10/req_009','discord10/req_010']]
    assert set(original)=={capture for group in historical_groups for capture in group}
    tasks=list(dict.fromkeys(r['task'] for r in historical if not r['task'].startswith('level')))
    assert len(tasks)==11
    recent=json.loads((RECENT/'inputs/tokenized.json').read_text())
    needed=64-len(tasks)
    tasks.extend(r['task'] for r in recent[:needed])
    assert len(tasks)==len(set(tasks))==64
    chosen=set(tasks);records=[];groups=[]
    for group in historical_groups:
        groups.append(group)
        for capture in group:
            row=dict(original[capture])
            path=HIST/'replay'/(capture.replace('/','_')+'.json')
            raw=path.read_bytes();result=json.loads(raw)
            assert result['prompt_hash']==row['prompt_hash'] and result['complete']
            expected_group=result.get('group_prompts',1)
            assert expected_group==len(group)
            # The old single-prompt driver defaults to teacher width 4.
            # The grouped driver records its width explicitly as 8.
            capacity=max(1024,row['output_tokens'])+result.get('teacher_width',4)
            row.update(source_root=str(HIST),included=row['task'] in chosen,
                reuse_path=str(path),reuse_sha256=hashlib.sha256(raw).hexdigest(),
                cache_capacity=capacity,first_logits_name='dense_logits.pt',
                first_logits_row=0,source_kind='historical',task_order=tasks.index(row['task']) if row['task'] in chosen else None)
            records.append(row)
    # Keep paired row positions. One extra prompt is a numerical batch partner,
    # is explicitly excluded from the 64-task pool, and never enters rankings.
    for start in range(0,needed,2):
        pair=recent[start:start+2];groups.append([r['capture'] for r in pair])
        for source in pair:
            row=dict(source);path=RECENT/'replay'/(row['capture'].replace('/','_')+'.json')
            reuse={}
            if path.exists():
                raw=path.read_bytes();result=json.loads(raw)
                assert result['prompt_hash']==row['prompt_hash'] and result['complete']
                reuse=dict(reuse_path=str(path),reuse_sha256=hashlib.sha256(raw).hexdigest())
            row.update(source_root=str(RECENT),included=row['task'] in chosen,
                cache_capacity=1032,first_logits_name='first_logits.pt',first_logits_row=None,
                source_kind='recent',task_order=tasks.index(row['task']) if row['task'] in chosen else None,**reuse)
            records.append(row)
    assert len({r['capture'] for r in records})==len(records)
    included=[r for r in records if r['included']]
    reused=[r for r in included if r.get('reuse_path')]
    assert len(included)==72 and len(reused)==21
    for row in records:
        assert (Path(row['source_root'])/'captures'/row['capture']/'layer_63.pt').exists()
        assert row['geometry']['16']['maximum']<=16
    output=dict(tasks=tasks,groups=groups,records=records)
    destination=ROOT/'inputs/records.json';destination.parent.mkdir(exist_ok=True)
    if destination.exists():assert json.loads(destination.read_text())==output
    else:destination.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(planned_tasks=64,planned_included_steps=len(included),
        prior_steps=len(reused),prior_tasks=len({r['task'] for r in reused}),
        batch_partner_states=sum(not r['included'] for r in records),
        records_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),tasks=tasks,
        records=[{k:r.get(k) for k in ('task','capture','included','task_order','source_kind','source_root','reuse_path',
            'reuse_sha256','prompt_hash','prompt_tokens','cache_capacity')} for r in records])
    (ROOT/'inputs/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('tasks','records')}))


if __name__=='__main__':main()
