"""Measured rankings and explicitly outcome-selected calibration examples."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TARGET=(32,.25)


def table(headers,rows):
    clean=lambda x:str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(clean,row))+' |' for row in rows])


def pct(value):
    return f'{value:.2%}' if pd.notna(value) else 'undefined'


def rank_table(ranks,cohort,metric='target_equal'):
    frame=ranks[(ranks.cohort==cohort)&(ranks.metric==metric)].sort_values(['rank','A','alpha'])
    return table(['Rank','A','alpha','Agreement','Eligible tasks'],
        [(int(r['rank']) if pd.notna(r['rank']) else 'undefined',int(r['A']),f"{r['alpha']:g}",pct(r['mean']),int(r['count']))
         for _,r in frame.iterrows()])


def plot_rankings(ranks,cohorts,titles,path):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(9.3,7.1),sharex=True)
    for ax,cohort,title in zip(axes,cohorts,titles):
        frame=ranks[(ranks.cohort==cohort)&(ranks.metric=='target_equal')].sort_values(['rank','A','alpha'])
        if frame.empty or frame['count'].max()==0:
            ax.text(.5,.5,'Insufficient eligible tasks',ha='center',va='center',transform=ax.transAxes)
            ax.set_title(title);ax.set_axis_off();continue
        labels=[f'({int(r.A)}, {r.alpha:g})' for r in frame.itertuples()]
        colors=['#9B2639' if (int(r.A),r.alpha)==TARGET else '#526D82' for r in frame.itertuples()]
        y=np.arange(len(frame));ax.barh(y,frame['mean'],height=.68,color=colors)
        ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,1)
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_axisbelow(True)
        ax.grid(axis='x',color='#DDDDDD',linewidth=.5)
        for yi,mean in zip(y,frame['mean']):ax.text(1.02,yi,f'{mean:.1%}',ha='left',va='center',fontsize=7,clip_on=False)
        ax.set_xlabel('Action type and target index agreement');ax.set_ylabel('(A, alpha)')
        ax.set_title(title+f"\n{int(frame['count'].max())} eligible tasks",pad=9)
    fig.subplots_adjust(left=.11,right=.92,bottom=.09,top=.92,wspace=.65)
    for extension in ('png','pdf','svg'):fig.savefig(path.with_suffix('.'+extension),dpi=220,bbox_inches='tight')
    plt.close(fig)


def main():
    status=json.loads((ROOT/'results/analysis_manifest.json').read_text())
    searches=json.loads((ROOT/'results/subset_search.json').read_text())
    metadata=json.loads((ROOT/'results/task_metadata.json').read_text());by_task={r['task']:r for r in metadata}
    ranks=pd.read_csv(ROOT/'results/rankings.csv');count=status['completed_tasks']
    settings=status['settings']
    lines=['# Offline action calibration with 64-token pages',
        f"Protocol revision: 2026-11-12. **{count}/64 tasks completed**. "
        f"Dense indexed actions are available for **{status['dense_index_eligible_tasks']}** tasks. "
        f"Status: {'complete' if status['complete'] else 'interim, measurement continues'}. "
        'This experiment searches for calibration examples that favor (32,0.25). Such outcome-selected subsets are exploratory, '
        'and their ranks are not unbiased evidence of general superiority.',
        '## 1. Configuration and metric',
        'Qwen3-32B BF16 runs on spark00 with all 64 layers, greedy decoding, thinking enabled, and a 1024-token response limit. '
        'Each task contributes one archived WebVoyager state. This is not a full browser trajectory or an online task-success experiment. '
        'Images are removed and text retained. The 64-task pool and its ordering were frozen before measurement. '
        'The source step must have an indexed action whose target appears in the DOM. Source task success is not a requirement.',
        'The only profile uses **page size 64, maximum chunk size 32, soft merge target 6, and nominal distinct-page budget 12.5%**. '
        'All input tokens remain represented. Mandatory prefix, recent and generated reservations follow the existing implementation. '
        'A ranges over {1,2,4,8,16,32,64,128}, and alpha over {0.125,0.25,0.5}. The 24 sparse variants and one dense reference '
        'share dense prefill, then generate independent states. Fixed pairs give 50 decoder rows. '
        'Earlier two-profile results are preserved separately and are excluded from this report.',
        'Joint agreement requires the complete action-type sequence and flattened target-index sequence to match the paired dense response. '
        'Other arguments, such as typed text, can differ. Complete-action equality separately requires all parsed arguments to match. '
        'Unfinished or invalid sparse responses count as mismatches. Dense responses without a completed, parseable indexed action '
        'are excluded only from the joint metric and remain in the exports. Dense agreement does not establish action correctness.',
        '[Protocol](protocol.md), [settings](settings.json), and [frozen task list](inputs/candidate_manifest.json).',
        '## 2. Overall and fixed-sample rankings']
    summary=[]
    for cohort in ('primary16','primary32','measured_pool','exploratory16','exploratory32'):
        frame=ranks[(ranks.cohort==cohort)&(ranks.metric=='target_equal')]
        if frame.empty:continue
        target=frame[(frame.A==32)&(frame.alpha==.25)].iloc[0]
        winners=frame[frame['rank']==1]
        summary.append((cohort,int(target['count']),pct(target['mean']),
            int(target['rank']) if pd.notna(target['rank']) else 'undefined',
            ', '.join(f'({int(r.A)}, {r.alpha:g})' for r in winners.itertuples())))
    lines.append(table(['Cohort','Eligible tasks','(32,0.25)','Rank / 24','Winner(s)'],summary))
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    plot_rankings(ranks,['primary32','measured_pool'],['(a) Fixed first 32 tasks','(b) All completed tasks'],figures/'fixed_and_all_rankings')
    lines.append('![Fixed and complete observed pool rankings](figures/fixed_and_all_rankings.png)')
    for cohort in ('primary16','primary32','measured_pool'):
        if cohort not in set(ranks.cohort):continue
        lines.extend([f'### {cohort}',rank_table(ranks,cohort)])
    lines.extend(['## 3. Outcome-selected subsets',
        'Screening starts after 16 completed steps and repeats after each completed pair through all 64. '
        'Finding a favorable subset does not stop measurement. The optimizer selects exactly 16 or 32 distinct tasks with eligible dense actions. '
        'It maximizes the minimum match-count margin of (32,0.25) over all 23 rivals, then maximizes its own match count. '
        'Positive margin means unique first place, zero means first including ties, and a negative exact optimum rules out first place '
        'for any subset of that size within the measured eligible pool.'])
    plot_rankings(ranks,['exploratory16','exploratory32'],['(a) Selected 16-task set','(b) Selected 32-task set'],figures/'selected_rankings')
    lines.append('![Outcome-selected subset rankings](figures/selected_rankings.png)')
    for size in (16,32):
        search=searches[str(size)];lines.extend([f'### Selected {size}-task set',f"Search outcome: **{search['outcome']}**."])
        if 'task_ids' not in search:continue
        tasks=search['task_ids'];reference=search['random_subset_reference']
        lines.append(f"The target matches {search['target_matches']}/{size} dense indexed actions. Its margin over the strongest rival is "
            f"{search['minimum_margin_matches']} matches. Among 10,000 uniform {size}-task subsets of the same observed eligible pool, "
            f"it ranks uniquely first in {reference['unique_first_fraction']:.2%}, and first including ties in "
            f"{reference['first_including_ties_fraction']:.2%}. These frequencies describe this pool, not unseen data.")
        lines.append(rank_table(ranks,f'exploratory{size}'))
        other=[]
        for metric in ('type_equal','action_equal'):
            f=ranks[(ranks.cohort==f'exploratory{size}')&(ranks.metric==metric)&(ranks.A==32)&(ranks.alpha==.25)].iloc[0]
            other.append((metric,pct(f['mean']),int(f['rank']),int(f['count'])))
        lines.append(table(['Additional metric','(32,0.25)','Rank / 24','Eligible tasks'],other))
        websites=Counter(by_task[t]['website'] for t in tasks)
        lines.append('Website composition: '+', '.join(f'{site}: {n}' for site,n in sorted(websites.items()))+'.')
        lines.append(table(['Task','Archived step','Prompt tokens','Dense action types','Dense target indices'],
            [(t,by_task[t]['step'],by_task[t]['prompt_tokens'],str(by_task[t]['dense_types']),str(by_task[t]['dense_indices'])) for t in tasks]))
        (ROOT/'results'/f'selected{size}_tasks.json').write_text(json.dumps([by_task[t] for t in tasks],ensure_ascii=False,indent=2)+'\n')
    lines.append('## 4. Rolling search history')
    history=[]
    for checkpoint in sorted((ROOT/'results/checkpoints').glob('n*')):
        snapshot=json.loads((checkpoint/'analysis_manifest.json').read_text())
        found=json.loads((checkpoint/'subset_search.json').read_text())
        for size,result in found.items():
            history.append((snapshot['completed_tasks'],snapshot['dense_index_eligible_tasks'],size,result['outcome'],
                            result.get('minimum_margin_matches','undefined')))
    lines.append(table(['Completed tasks','Eligible tasks','Subset size','Outcome','Worst-rival margin'],history))
    lines.extend(['## 5. Validation and interpretation',
        'The optimized memory gather is checked bit for bit against direct Torch indexing, including unused slots, '
        'prefix/generated boundaries, partial final pages and CUDA graph execution. A matched complete-model control '
        'compares the first 32 generated tokens using the original and optimized memory operations at identical batch size. '
        'This is a finite regression check, not exhaustive proof that every later long response is identical. '
        'The source parser, score kernels, packing and selection are pinned. Prior controls showed BF16 free generation '
        'can depend on batch membership, so this run uses a fixed new schedule and reruns its earliest tasks.',
        'The selected sets deliberately favor a requested configuration. They answer whether such a subset exists in this pool. '
        'Neither their complementary tasks nor the larger measured pool are untouched validation data. '
        'A practical parameter choice should use a frozen representative calibration distribution and a declared objective, '
        'then be evaluated on independent tasks. No result here establishes online task accuracy.',
        '[Full rankings](results/rankings.csv), [per-task metrics](results/per_task.csv), '
        '[action comparisons](results/action_comparisons.json), [subset membership](results/subset_membership.csv), '
        '[search results](results/subset_search.json), and [provenance](results/analysis_manifest.json).'])
    control=ROOT/'control/model_validation.json'
    if control.exists():
        validation=json.loads(control.read_text());assert validation['passed']
        lines.append(f"All {len(validation['checks'])} full-model branch-prefix comparisons passed. "
                     '[Control details](control/model_validation.json).')
    messages={r['task']:r['messages'] for r in json.loads((ROOT/'inputs/candidate_messages.json').read_text())['records']}
    tokenized={r['task']:r for r in json.loads((ROOT/'inputs/tokenized.json').read_text())}
    datasets=ROOT/'datasets';datasets.mkdir(exist_ok=True)
    lines.append('## 6. Reusable selected examples')
    for size,search in searches.items():
        if 'task_ids' not in search:continue
        path=datasets/f'selected{size}.jsonl';records=[]
        for task in search['task_ids']:
            records.append(dict(task=task,question=by_task[task]['question'],source_snapshot=by_task[task]['original_snapshot'],
                messages=messages[task],input_ids=tokenized[task]['input_ids'],prompt_hash=tokenized[task]['prompt_hash'],
                dense_actions=by_task[task]['dense_actions'],selected_using_grid_outcomes=True))
        assert len(records)==int(size) and len({r['task'] for r in records})==int(size)
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        path.with_suffix('.metadata.json').write_text(json.dumps(dict(settings=settings,selected_using_grid_outcomes=True,
            tasks=search['task_ids'],data_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
        lines.append(f'[{size} selected prompt records](datasets/{path.name}). These records were selected using all grid outcomes.')
    if (ROOT/'diagnostics/metric_comparison_n054.md').exists():
        lines.extend(['## Diagnostic note on format dependence',
            'The declared joint metric also requires completion and acceptance by the existing JSON/index parser. '
            'An audit at the 54-step checkpoint found explicit matching actions that were rejected because they used XML '
            'or alternative target-field names. These failures must not all be interpreted as wrong element selection. '
            '[Metric comparison and raw examples](diagnostics/metric_comparison_n054.md).'])
    (ROOT/'report.en.md').write_text('\n\n'.join(lines)+'\n')
    print(ROOT/'report.en.md')


if __name__=='__main__':main()
