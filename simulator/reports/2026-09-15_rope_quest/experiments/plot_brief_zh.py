"""Chinese versions of the brief's two figures; source data are read-only.

Run: python3 research/2026-09-15_rope_quest/experiments/plot_brief_zh.py
Requires NumPy, Matplotlib, and a Chinese font available on this Mac.
Does not import or overwrite the English plotting scripts/figures.
"""
from pathlib import Path
import csv
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Rectangle, FancyArrowPatch, Arc

ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
if not FONT_PATH.exists():
    FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
fontManager.addfont(str(FONT_PATH))
CHINESE_FONT = FontProperties(fname=str(FONT_PATH)).get_name()
plt.rcParams.update({
    "font.family": [CHINESE_FONT, "DejaVu Sans"], "font.size": 12,
    "mathtext.fontset": "dejavusans", "axes.unicode_minus": False,
    "text.color": "#28343B", "axes.labelcolor": "#28343B",
    "xtick.color": "#28343B", "ytick.color": "#28343B",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "normal",
})
BLUE, PURPLE, RUST = "#4E738A", "#776284", "#AD6952"
GRAY, INK, LIGHT = "#9DA8AC", "#28343B", "#E9EFF1"


def arrow(ax, endpoint, color, lw=2):
    ax.add_patch(FancyArrowPatch((0, 0), endpoint, arrowstyle="-|>",
                 mutation_scale=14, lw=lw, color=color, zorder=3))


def frame(ax, lim):
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axhline(0, color=GRAY, lw=.85, zorder=0)
    ax.axvline(0, color=GRAY, lw=.85, zorder=0)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.tick_params(length=0, pad=3, labelsize=10)
    ax.text(lim-.03, .09, "$x_1$", ha="right", fontsize=11)
    ax.text(.08, lim-.04, "$x_2$", va="top", fontsize=11)


def box_bound(keys, query):
    return np.maximum(query*keys.min(0), query*keys.max(0)).sum()


def geometry():
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.55))
    fig.subplots_adjust(left=.025, right=.982, bottom=.24, top=.79, wspace=.19)

    ax = axes[0]
    ax.set_aspect("equal"); ax.set_xlim(-.25, 1.38); ax.set_ylim(-.25, 1.38)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axhline(0, color=GRAY, lw=.85); ax.axvline(0, color=GRAY, lw=.85)
    ax.set_xticks([0]); ax.set_yticks([0, 1]); ax.tick_params(length=0, pad=3, labelsize=10)
    ax.text(1.35, .055, "$x_1$", ha="right", fontsize=11)
    ax.text(.055, 1.35, "$x_2$", va="top", fontsize=11)
    angles = np.arange(3)*np.pi/4
    keys_a = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    arc = np.linspace(0, np.pi/2, 200)
    ax.plot(np.cos(arc), np.sin(arc), color=GRAY, lw=1.2, ls=(0,(3,3)), zorder=0)
    for key in keys_a:
        arrow(ax, key, BLUE, 1.9)
        ax.scatter(*key, s=35, color=BLUE, zorder=5)
    ax.annotate("位置 0\n$(1,0)$", keys_a[0], xytext=(0,-12), textcoords="offset points", ha="center", va="top", fontsize=11)
    ax.annotate("位置 1", keys_a[1], xytext=(5,3), textcoords="offset points", ha="left", va="bottom", fontsize=11)
    ax.annotate("位置 2\n$(0,1)$", keys_a[2], xytext=(-8,8), textcoords="offset points", ha="center", va="bottom", fontsize=11)
    ax.add_patch(Arc((0,0), .70, .70, theta1=0, theta2=45, color=RUST, lw=1.4))
    ax.add_patch(Arc((0,0), .86, .86, theta1=45, theta2=90, color=RUST, lw=1.4))
    ax.text(.46,.15,"$\\theta$", color=RUST, fontsize=13)
    ax.text(.17,.50,"$\\theta$", color=RUST, fontsize=13)
    ax.set_title("(a) 同一内容随位置旋转", fontsize=12.5, pad=23)

    ax=axes[1]; frame(ax,1.75)
    keys_b=np.array([[1.,-1.],[-1.,1.]])
    query_b=np.array([1.,1.])
    ax.add_patch(Rectangle((-1,-1),2,2,facecolor=LIGHT,edgecolor=GRAY,lw=1.2,zorder=-1))
    ax.plot(keys_b[:,0],keys_b[:,1],color=GRAY,lw=1,ls=(0,(3,3)),zorder=1)
    ax.scatter(keys_b[:,0],keys_b[:,1],s=50,color=BLUE,zorder=5)
    arrow(ax,query_b,PURPLE)
    ax.scatter([1],[1],s=73,marker="D",facecolor="white",edgecolor=RUST,linewidth=1.8,zorder=6)
    ax.annotate("$(1,-1)$",(1,-1),xytext=(0,-16),textcoords="offset points",ha="center",fontsize=11)
    ax.annotate("$(-1,1)$",(-1,1),xytext=(-3,11),textcoords="offset points",ha="center",fontsize=11)
    ax.annotate("无对应 token\n$(1,1)$",(1,1),xytext=(3,13),textcoords="offset points",ha="center",color=RUST,fontsize=10.5)
    ax.text(.02,.51,"$q=(1,1)$",color=PURPLE,rotation=45,fontsize=11,ha="center")
    ax.set_title("(b) 包围盒包含虚构角点",fontsize=12.5,pad=23)

    ax=axes[2]; frame(ax,1.75)
    angle=np.pi/4
    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    keys_c=keys_b@rot.T; query_c=query_b@rot.T
    assert np.max(np.abs(keys_b@query_b-keys_c@query_c))<1e-14
    assert abs(box_bound(keys_b,query_b)-2)<1e-14
    assert abs(box_bound(keys_c,query_c))<1e-14
    ax.plot([-np.sqrt(2),np.sqrt(2)],[0,0],color=GRAY,lw=5,solid_capstyle="round",zorder=2)
    ax.scatter(keys_c[:,0],keys_c[:,1],s=50,color=BLUE,zorder=5)
    arrow(ax,query_c,PURPLE,2.2)
    ax.annotate("$(\\sqrt{2},0)$",keys_c[0],xytext=(-3,-18),textcoords="offset points",ha="center",fontsize=11)
    ax.annotate("$(-\\sqrt{2},0)$",keys_c[1],xytext=(3,-18),textcoords="offset points",ha="center",fontsize=11)
    ax.text(.17,1.06,"$q'=(0,\\sqrt{2})$",color=PURPLE,fontsize=11)
    ax.text(0,-.70,"包围盒退化为一条线",ha="center",fontsize=11)
    ax.set_title("(c) Q/K 同时旋转 45°",fontsize=12.5,pad=23)

    centers=[ax.get_position().x0+ax.get_position().width/2 for ax in axes]
    captions=["同一内容，每个位置转 45°\n向量长度保持不变",
              "真实点积：0，0\nQuest 上界：2",
              "真实点积：0，0（不变）\nQuest 上界：0"]
    for cx,caption in zip(centers,captions):
        fig.text(cx,.103,caption,ha="center",va="center",fontsize=11.5,linespacing=1.65)
    fig.text(.5,.96,"RoPE 的位置旋转与 Quest 的包围盒误差",ha="center",va="top",fontsize=15)
    path=ROOT/"mechanism/rope_geometry_tutorial_zh.png"
    fig.savefig(path,dpi=250,facecolor="white")
    plt.close(fig)
    return path


def frequency():
    source=ROOT/"results/summary_betas.csv"
    original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(newline="") as f:
        betas=list(csv.DictReader(f))
    bands=[f"band_{j}_{j+7}" for j in range(0,64,8)]
    labels=[f"{j}–{j+7}" for j in range(0,64,8)]
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.55))
    fig.subplots_adjust(left=.073,right=.988,top=.85,bottom=.265,wspace=.26)
    configs=[
        (axes[0],[("original","原始 RoPE",BLUE),("collapse_all","去掉块内相位差",RUST)],
         "(a) 去掉相位差，中频峰值消退"),
        (axes[1],[("phase_speed_025","1/4 倍速度","#658675"),("original","原始速度",BLUE),("phase_speed_4","4 倍速度",PURPLE)],
         "(b) 改变旋转速度，峰值随之移动"),
    ]
    for ax,conditions,title in configs:
        for mode,label,color in conditions:
            rows=[next(r for r in betas if r["mode"]==mode and r["metric"]==band) for band in bands]
            estimate=np.array([float(r["endpoint_exponent"]) for r in rows])
            low=np.array([float(r["endpoint_exponent_ci_low"]) for r in rows])
            high=np.array([float(r["endpoint_exponent_ci_high"]) for r in rows])
            ax.errorbar(np.arange(8),estimate,yerr=np.stack([estimate-low,high-estimate]),
                        fmt="o-",color=color,label=label,ms=4.2,lw=1.5,capsize=2)
        ax.set_xticks(np.arange(8),labels,rotation=35,fontsize=10.5)
        ax.set_xlabel("RoPE 维度对编号（由快到慢）",fontsize=11.5,labelpad=5)
        ax.set_ylim(.13,.36)
        ax.set_ylabel("端点增长指数（8–256 token）",fontsize=11.5,labelpad=7)
        ax.tick_params(axis="y",labelsize=10.5)
        ax.set_title(title,loc="left",fontsize=12.4,pad=14)
        ax.grid(axis="y",color="#E7E7E7",lw=.7); ax.set_axisbelow(True)
        ax.legend(frameon=False,loc="upper right",fontsize=10.2,handlelength=2)
    fig.text(.5,.052,"指标为未按 query 加权的坐标极差（range），并非加权 width；误差条为 95% 任务级 bootstrap 区间。",
             ha="center",va="center",fontsize=10.5,color=INK)
    path=ROOT/"figures/real_frequency_interventions_zh.png"
    fig.savefig(path,dpi=250,facecolor="white")
    plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==original_hash
    return path


if __name__=="__main__":
    print("Chinese font:",CHINESE_FONT)
    for path in [geometry(),frequency()]:
        print("Saved",path)
