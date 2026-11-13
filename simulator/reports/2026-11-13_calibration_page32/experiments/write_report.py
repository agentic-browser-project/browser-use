"""English report for the declared strict metric and outcome-selected sets."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TARGETS=((32,.25),(16,.25),(8,.25))


def table(headers,rows):
    clean=lambda x:str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(clean,row))+' |' for row in rows])


def ranking_table(frame):
    return table(['Rank','A','alpha','Task-averaged joint agreement','Eligible tasks'],
        [(int(r['rank']) if pd.notna(r['rank']) else 'undefined',int(r['A']),f"{r['alpha']:g}",
          f"{r['mean']:.2%}" if pd.notna(r['mean']) else 'undefined',int(r['count'])) for _,r in frame.iterrows()])


def plot(ranks,cohorts,titles,filename):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(9.8,7.0))
    available=ranks[(ranks.cohort.isin(cohorts))&(ranks.metric=='target_equal')]['mean'].dropna()
    limit=min(1,max(.1,np.ceil((available.max() if len(available) else .1)*10)/10+.05))
    palette={(32,.25):'#9B2639',(16,.25):'#496F88',(8,.25):'#AD8B43'}
    for ax,cohort,title in zip(axes,cohorts,titles):
        frame=ranks[(ranks.cohort==cohort)&(ranks.metric=='target_equal')].sort_values(['rank','A','alpha'])
        if frame.empty or frame['count'].max()==0:
            ax.text(.5,.5,'Insufficient eligible tasks',ha='center',transform=ax.transAxes);ax.set_axis_off();continue
        configs=[(int(r.A),r.alpha) for r in frame.itertuples()];y=np.arange(len(frame))
        ax.barh(y,frame['mean'],height=.68,color=[palette.get(c,'#B2BAC0') for c in configs])
        ax.set_yticks(y,[f'({a}, {b:g})' for a,b in configs]);ax.invert_yaxis();ax.set_xlim(0,limit)
        for tick,c in zip(ax.get_yticklabels(),configs):
            if c in TARGETS:tick.set_fontweight('bold');tick.set_color(palette[c])
        for yi,value in zip(y,frame['mean']):
            ax.text(1.02,yi,f'{value:.1%}',transform=ax.get_yaxis_transform(),va='center',fontsize=7,clip_on=False)
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_axisbelow(True)
        ax.grid(axis='x',color='#DDDDDD',linewidth=.5)
        ax.set_title(title+f"\n{int(frame['count'].max())} eligible tasks")
        ax.set_xlabel('Task-averaged joint agreement');ax.set_ylabel('(A, alpha)')
    fig.subplots_adjust(left=.11,right=.92,wspace=.65,bottom=.09,top=.92)
    for extension in ('png','pdf','svg'):fig.savefig(filename.with_suffix('.'+extension),dpi=220,bbox_inches='tight')
    plt.close(fig)


def main():
    status=json.loads((ROOT/'results/analysis_manifest.json').read_text())
    searches=json.loads((ROOT/'results/subset_search.json').read_text())
    ranks=pd.read_csv(ROOT/'results/rankings.csv');tasks=pd.read_csv(ROOT/'results/per_task.csv')
    lines=['# Page32 calibration with legacy results and 64 distinct tasks',
        f"Protocol revision: 2026-11-13. **{status['completed_tasks']}/64 tasks** have complete 24-setting grids, "
        f"covering **{status['completed_steps']}/72 included states**. Joint agreement is defined for "
        f"**{status['eligible_tasks']} tasks**. Status: {'complete' if status['complete'] else 'measurement continues'}.",
        '## 1. Configuration, reuse and metric',
        'Qwen3-32B BF16 runs on spark00 with page size 32, maximum chunk size 16, soft merge target 6 and nominal '
        'distinct-page budget 12.5%. Generation is greedy, thinking is enabled, and the response limit is 1024 tokens. '
        'The reported grid crosses A in {1,2,4,8,16,32,64,128} with alpha in {0.125,0.25,0.5}. '
        'No browser actions are executed.',
        'The pool retains 21 previously completed old-profile states from 13 tasks. Fifteen settings overlap the current grid. '
        'Those stored outputs are retained, and nine settings are supplemented. Legacy model row positions, batch partners, '
        'score shapes and cache capacities are preserved. Stored legacy cap16 and dense continuations are checked token for token '
        'before any merged result is accepted. Additional old alpha=0.375 rows are numerical controls and are absent from the reported grid. '
        'The remaining 51 distinct tasks follow the previously frozen candidate order. There are 72 included states from 64 tasks. '
        'Three numerical batch partners are excluded from all task counts and rankings.',
        'The primary metric compares action types and target indices with the paired dense response. '
        'Integer IDs and their integer-string representations are equivalent, so 8117 and "8117" match. '
        'The former raw comparison remains available as target_equal_raw. This correction applies to every configuration. '
        'It requires a finished output accepted by the existing action JSON parser. It does not normalize XML or alternative target-field names. '
        'Consequently it combines completion, format acceptance and action fidelity. It is not pure semantic element accuracy or online success. '
        'Unfinished or unparseable sparse outputs count as mismatches. Dense states without parseable indexed actions are excluded only '
        'from the joint metric. Average eligible steps within each task before giving tasks equal weight.',
        '[Protocol](protocol.md), [settings](settings.json), and [input/provenance inventory](inputs/manifest.json).',
        '## 2. Main target configurations']
    overview=[]
    for cohort in ('measured_pool','primary16','primary32','exploratory16','exploratory32'):
        for a,b in TARGETS:
            f=ranks[(ranks.cohort==cohort)&(ranks.metric=='target_equal')&(ranks.A==a)&(ranks.alpha==b)]
            if f.empty:continue
            r=f.iloc[0]
            overview.append((cohort,a,b,f"{r['mean']:.2%}" if pd.notna(r['mean']) else 'undefined',
                int(r['rank']) if pd.notna(r['rank']) else 'undefined',int(r['count'])))
    lines.append(table(['Cohort','A','alpha','Joint agreement','Rank / 24','Eligible tasks'],overview))
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    plot(ranks,['primary32','measured_pool'],['(a) Fixed first 32 tasks','(b) All complete tasks'],figures/'fixed_and_all_rankings')
    lines.append('![Fixed and full-pool rankings](figures/fixed_and_all_rankings.png)')
    for cohort in ('measured_pool','primary16','primary32'):
        frame=ranks[(ranks.cohort==cohort)&(ranks.metric=='target_equal')].sort_values(['rank','A','alpha'])
        if not frame.empty:lines.extend([f'### {cohort}',ranking_table(frame)])
    lines.extend(['## 3. Exploratory subset search',
        'Search begins after 16 distinct tasks have complete grids and continues through all 64 tasks. '
        'For each subset size, first maximize the worst-rival margin of (32,0.25). At that optimum, minimize the worse '
        'rank of (16,0.25) and (8,0.25), then their rank sum, then maximize the (32,0.25) score. '
        'Integer-scaled task means preserve exact comparisons when a task contains multiple eligible steps. '
        'A positive optimal margin establishes unique first place, zero permits tied first, and a negative optimum rules out first '
        'within the eligible measured pool for that size. A negative-margin solution is not claimed to minimize the primary competition rank.'])
    plot(ranks,['exploratory16','exploratory32'],['(a) Selected 16-task set','(b) Selected 32-task set'],figures/'selected_rankings')
    lines.append('![Outcome-selected subsets](figures/selected_rankings.png)')
    for size,search in searches.items():
        lines.extend([f'### Selected {size}-task set',f"Search outcome: **{search['outcome']}**."])
        if 'task_ids' not in search:continue
        lines.append(f"The primary worst-rival margin is {search['minimum_margin_fraction']:.2%}. "
                     f"Its joint score is {search['target_score']:.2%} and its competition rank is {search['target_rank']}.")
        frame=ranks[(ranks.cohort==f'exploratory{size}')&(ranks.metric=='target_equal')].sort_values(['rank','A','alpha'])
        lines.append(ranking_table(frame))
        lines.append('Selected tasks: '+', '.join(search['task_ids'])+'.')
        rows=[]
        for task in search['task_ids']:
            f=tasks[(tasks.task==task)&(tasks.A==32)&(tasks.alpha==.25)].iloc[0]
            rows.append((task,int(f.included_steps),int(f.joint_eligible_steps),f'{f.target_equal:.2%}'))
        lines.append(table(['Task','Included steps','Eligible joint-metric steps','(32,0.25) task score'],rows))
    lines.append('## 4. Rolling history')
    history=[]
    for checkpoint in sorted((ROOT/'results/checkpoints').glob('n*')):
        stage=json.loads((checkpoint/'analysis_manifest.json').read_text())
        searches_at_stage=json.loads((checkpoint/'subset_search.json').read_text())
        for size,result in searches_at_stage.items():
            history.append((stage['completed_tasks'],stage['eligible_tasks'],size,result['outcome'],
                result.get('target_rank','undefined'),str(result.get('secondary_ranks',[]))))
    lines.append(table(['Completed tasks','Eligible tasks','Subset size','Outcome','Primary rank','Secondary ranks'],history))
    lines.extend(['## 5. Interpretation and artifacts',
        'These subsets are selected using measured outcomes to favor explicitly named configurations. Their rankings answer '
        'an existence question within this pool and do not establish superiority on unseen tasks. The complementary tasks '
        'are also inspected data. Keep full-pool and frozen-prefix results alongside every selected set.',
        '[Full rankings](results/rankings.csv), [task means and denominators](results/per_task.csv), '
        '[per-state outcomes](results/per_step.csv), [selected membership](results/subset_membership.csv), '
        '[optimization results](results/subset_search.json), and [analysis provenance](results/analysis_manifest.json).'])
    (ROOT/'report.en.md').write_text('\n\n'.join(lines)+'\n')
    print(ROOT/'report.en.md')


if __name__=='__main__':main()
