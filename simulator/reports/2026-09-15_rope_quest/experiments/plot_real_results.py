from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#283238','text.color':'#283238','axes.titleweight':'normal','pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':200})
BLUE='#476f87';ORANGE='#b86d4d';GREEN='#658675';GRAY='#7b7e88';PURPLE='#82708e'
curves=list(csv.DictReader((ROOT/'results/summary_curves.csv').open()))
betas=list(csv.DictReader((ROOT/'results/summary_betas.csv').open()))

def curve(mode,metric,lengths):
    rows=[next(r for r in curves if r['mode']==mode and r['metric']==metric and int(r['length'])==l) for l in lengths]
    return tuple(np.array([float(r[k]) for r in rows]) for k in ['estimate','estimate_ci_low','estimate_ci_high'])

def save(fig,name):
    fig.savefig(ROOT/f'figures/{name}.png',bbox_inches='tight')
    fig.savefig(ROOT/f'figures/{name}.pdf',bbox_inches='tight');plt.close(fig)

lengths=[8,16,32,64,128,256,512,1024]
fig,axs=plt.subplots(1,2,figsize=(10.8,3.8),layout='constrained')
for ax,metric,title in zip(axs,['pooled_width','pooled_gap'],['(a) Query-weighted box width','(b) Bound minus true maximum']):
    for mode,label,col in [('original','Original RoPE',BLUE),('collapse_all','Within-chunk phase removed',ORANGE)]:
        y,lo,hi=curve(mode,metric,lengths)
        ax.plot(lengths,y,'o-',color=col,label=label,ms=4,lw=1.7)
        ax.fill_between(lengths,lo,hi,color=col,alpha=.12,linewidth=0)
    ax.set_xscale('log',base=2);ax.set_xticks(lengths,[str(x) for x in lengths]);ax.set_xlabel('Chunk length (tokens)')
    ax.set_ylabel('Scaled logit units');ax.set_title(title,loc='left',pad=10)
    ax.grid(axis='y',color='#e7e7e7',lw=.7);ax.set_axisbelow(True)
axs[0].legend(frameon=False,loc='upper left',fontsize=9.5)
axs[1].annotate('At 256 tokens: 8.65% less gap\nSubstantial gap remains',xy=(256,17.8941),xytext=(13,24),arrowprops={'arrowstyle':'-','color':GRAY},fontsize=9.5)
save(fig,'real_width_and_gap')

bands=[f'band_{j}_{j+7}' for j in range(0,64,8)]
labels=[f'{j}–{j+7}' for j in range(0,64,8)]
fig,axs=plt.subplots(1,2,figsize=(11.2,4.1),layout='constrained')
for ax,conditions,title in [(axs[0],[('original','Original',BLUE),('collapse_all','Phase removed',ORANGE)],'(a) Removing phase removes the middle-band peak'),(axs[1],[('phase_speed_025','¼ speed',GREEN),('original','Original speed',BLUE),('phase_speed_4','4× speed',PURPLE)],'(b) Changing speed moves the peak')]:
    for mode,label,col in conditions:
        rs=[next(r for r in betas if r['mode']==mode and r['metric']==b) for b in bands]
        y=np.array([float(r['endpoint_exponent']) for r in rs]);lo=np.array([float(r['endpoint_exponent_ci_low']) for r in rs]);hi=np.array([float(r['endpoint_exponent_ci_high']) for r in rs])
        ax.errorbar(np.arange(8),y,yerr=np.stack([y-lo,hi-y]),fmt='o-',color=col,label=label,ms=4,lw=1.5,capsize=2)
    ax.set_xticks(np.arange(8),labels,rotation=35);ax.set_xlabel('Rotary pair index (fast → slow)')
    ax.set_ylim(.13,.36);ax.set_ylabel('Endpoint growth exponent, 8–256')
    ax.set_title(title,loc='left',fontsize=11,pad=10);ax.grid(axis='y',color='#e7e7e7',lw=.7);ax.set_axisbelow(True)
    ax.legend(frameon=False,loc='upper right',fontsize=9)
save(fig,'real_frequency_interventions')
print('Saved four figure files')

sr=list(csv.DictReader((ROOT/'results/selection_summary.csv').open()))
def sel(partition,mode,method,metric):
    return next(r for r in sr if r['partition']==partition and r['mode']==mode and r['method']==method and r['metric']==metric)
fig,axs=plt.subplots(1,2,figsize=(11.2,4.0),layout='constrained')
methods=[('quest','Quest',BLUE),('width_alpha025_anchor16','Length-calibrated',ORANGE),('exact_density_greedy','Exact-density greedy',GREEN)]
for i,(method,label,col) in enumerate(methods):
    rs=[sel(p,'original',method,'retained_mass') for p in ['native','fixed16']]
    y=np.array([float(r['estimate'])*100 for r in rs]);lo=np.array([float(r['estimate_ci_low'])*100 for r in rs]);hi=np.array([float(r['estimate_ci_high'])*100 for r in rs])
    axs[0].bar(np.arange(2)+(i-1)*.24,y,width=.22,color=col,label=label,yerr=np.stack([y-lo,hi-y]),error_kw={'lw':.9,'capsize':2,'ecolor':'#444444'})
    for x,v in zip(np.arange(2)+(i-1)*.24,y):axs[0].text(x,v+2.5,f'{v:.1f}',ha='center',fontsize=9)
axs[0].set_xticks([0,1],['Native variable chunks','Fixed 16 + final partial'])
axs[0].set_ylabel('Retained prefill attention mass (%)');axs[0].set_ylim(0,103)
axs[0].legend(frameon=False,loc='upper left',fontsize=8.5,ncol=1)
axs[0].set_title('(a) Same 2,048-token admission rule',loc='left',fontsize=11)
shifts=[137,997,4093]
for p,m,label,col,marker in [('native','quest','Quest, native chunks',BLUE,'o'),('fixed16','quest','Quest, fixed 16 + tail',ORANGE,'s'),('native','exact_density_greedy','Exact-density greedy',GREEN,'^')]:
    rs=[sel(p,f'common_shift_{s}',m,'selected_token_jaccard_vs_original') for s in shifts]
    y=np.array([float(r['estimate']) for r in rs]);lo=np.array([float(r['estimate_ci_low']) for r in rs]);hi=np.array([float(r['estimate_ci_high']) for r in rs])
    axs[1].errorbar(shifts,y,yerr=np.stack([y-lo,hi-y]),fmt=marker,color=col,label=label,capsize=3,ms=5,lw=1.5)
axs[1].set_xticks(shifts,[str(s) for s in shifts]);axs[1].set_xlim(-100,4350);axs[1].set_xlabel('Common RoPE shift δ (positions)')
axs[1].set_ylabel('Selected-token Jaccard vs. original');axs[1].set_ylim(.84,1.02)
axs[1].legend(frameon=False,loc='lower left',fontsize=9)
axs[1].set_title('(b) True attention is held fixed',loc='left',fontsize=11)
for ax in axs:ax.grid(axis='y',color='#e7e7e7',lw=.7);ax.set_axisbelow(True)
save(fig,'selection_controls')
