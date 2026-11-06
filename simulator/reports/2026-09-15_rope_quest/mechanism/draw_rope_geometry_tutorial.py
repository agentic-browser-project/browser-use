"""Three matched 2D examples for a beginner explanation of RoPE and Quest.

Run with /tmp/context-composition-venv/bin/python. Saves a PNG next to the source.
The b/c comparison uses exactly the same query and keys under a common rotation.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Arc

OUT = Path(__file__).resolve().parent
BLUE = "#4E738A"
PURPLE = "#776284"
RUST = "#AD6952"
GRAY = "#9DA8AC"
INK = "#28343B"
LIGHT = "#E9EFF1"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
                     "text.color":INK, "axes.labelcolor":INK,
                     "xtick.color":INK, "ytick.color":INK,
                     "mathtext.fontset":"dejavusans"})

def frame(ax, lim):
    ax.set_aspect("equal")
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axhline(0,color=GRAY,lw=.85,zorder=0)
    ax.axvline(0,color=GRAY,lw=.85,zorder=0)
    ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1])
    ax.tick_params(length=0,pad=3,labelsize=9)
    ax.text(lim-.03,.07,"$x_1$",ha="right",fontsize=11)
    ax.text(.08,lim-.04,"$x_2$",va="top",fontsize=11)

def arrow(ax, endpoint, color, lw=2):
    ax.add_patch(FancyArrowPatch((0,0),endpoint,arrowstyle="-|>",
                 mutation_scale=13,lw=lw,color=color,zorder=3))

def rotation(t):
    return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])

def bound(keys,q):
    lo,hi=keys.min(0),keys.max(0)
    return np.maximum(q*lo,q*hi).sum()

fig,axes=plt.subplots(1,3,figsize=(13.8,5.3))
fig.subplots_adjust(left=.035,right=.985,bottom=.235,top=.80,wspace=.18)

# (a): identical content, different token positions.
ax=axes[0]
ax.set_aspect("equal");ax.set_xlim(-.22,1.35);ax.set_ylim(-.22,1.35)
for sp in ax.spines.values():sp.set_visible(False)
ax.axhline(0,color=GRAY,lw=.85);ax.axvline(0,color=GRAY,lw=.85)
ax.set_xticks([0]);ax.set_yticks([0,1]);ax.tick_params(length=0,pad=3,labelsize=9)
ax.text(1.32,.055,"$x_1$",ha="right");ax.text(.055,1.32,"$x_2$",va="top")
angles=np.arange(3)*np.pi/4
keys_a=np.stack([np.cos(angles),np.sin(angles)],axis=1)
arc=np.linspace(0,np.pi/2,200)
ax.plot(np.cos(arc),np.sin(arc),color=GRAY,lw=1.2,ls=(0,(3,3)),zorder=0)
for i,k in enumerate(keys_a):
    arrow(ax,k,BLUE,lw=1.9)
    ax.scatter(*k,s=35,color=BLUE,zorder=5)
ax.annotate("$p=0$\n$k_0=(1,0)$",keys_a[0],xytext=(0,-14),textcoords="offset points",ha="center",va="top",fontsize=10)
ax.annotate("$p=1$",keys_a[1],xytext=(7,2),textcoords="offset points",ha="left",va="bottom",fontsize=10)
ax.annotate("$p=2$\n$k_2=(0,1)$",keys_a[2],xytext=(-8,8),textcoords="offset points",ha="center",va="bottom",fontsize=10)
ax.add_patch(Arc((0,0),.70,.70,theta1=0,theta2=45,color=RUST,lw=1.4))
ax.add_patch(Arc((0,0),.86,.86,theta1=45,theta2=90,color=RUST,lw=1.4))
ax.text(.45,.15,"$\\theta$",color=RUST,fontsize=13)
ax.text(.17,.48,"$\\theta$",color=RUST,fontsize=13)
ax.set_title("(a) Position rotates the same content",fontsize=12,pad=24)

# (b): box extrema from different tokens invent a corner.
ax=axes[1];frame(ax,1.68)
keys_b=np.array([[1.,-1.],[-1.,1.]])
q_b=np.array([1.,1.])
ax.add_patch(Rectangle((-1,-1),2,2,facecolor=LIGHT,edgecolor=GRAY,lw=1.2,zorder=-1))
ax.plot(keys_b[:,0],keys_b[:,1],color=GRAY,lw=1,ls=(0,(3,3)),zorder=1)
ax.scatter(keys_b[:,0],keys_b[:,1],s=50,color=BLUE,zorder=5)
arrow(ax,q_b,PURPLE)
ax.scatter([1],[1],s=73,marker="D",facecolor="white",edgecolor=RUST,linewidth=1.8,zorder=6)
ax.annotate("$k_1=(1,-1)$",(1,-1),xytext=(-2,-15),textcoords="offset points",ha="center",fontsize=10)
ax.annotate("$k_2=(-1,1)$",(-1,1),xytext=(-2,11),textcoords="offset points",ha="center",fontsize=10)
ax.annotate("Fictitious corner\n$(1,1)$",(1,1),xytext=(5,14),textcoords="offset points",ha="center",color=RUST,fontsize=10)
ax.text(.06,.50,"$q=(1,1)$",color=PURPLE,rotation=45,fontsize=10,ha="center")
ax.set_title("(b) The box adds an impossible corner",fontsize=12,pad=24)

# (c): the *same* 45-degree rotation of Q and K preserves exact logits.
ax=axes[2];frame(ax,1.68)
rot=rotation(np.pi/4)
keys_c=keys_b@rot.T;q_c=q_b@rot.T
assert np.max(np.abs(keys_b@q_b-keys_c@q_c))<1e-14
assert abs(bound(keys_b,q_b)-2)<1e-14
assert abs(bound(keys_c,q_c))<1e-14
ax.plot([-np.sqrt(2),np.sqrt(2)],[0,0],color=GRAY,lw=5,solid_capstyle="round",zorder=2)
ax.scatter(keys_c[:,0],keys_c[:,1],s=50,color=BLUE,zorder=5)
arrow(ax,q_c,PURPLE,lw=2.2)
ax.annotate("$k'_1=(\\sqrt{2},0)$",keys_c[0],xytext=(-7,-18),textcoords="offset points",ha="center",fontsize=10)
ax.annotate("$k'_2=(-\\sqrt{2},0)$",keys_c[1],xytext=(7,-18),textcoords="offset points",ha="center",fontsize=10)
ax.text(.15,1.07,"$q'=(0,\\sqrt{2})$",color=PURPLE,fontsize=11)
ax.text(0,-.57,"Bounding box is now a line",ha="center",color=INK,fontsize=10)
ax.set_title("(c) Rotate Q and K together by 45°",fontsize=12,pad=24)

centers=[ax.get_position().x0+ax.get_position().width/2 for ax in axes]
captions=[
    "$x=(1,0)$ and $\\theta=\\pi/4$ per position\nContent and vector length stay fixed",
    "True logits: $0,\\;0$\nQuest bound: $1+1=\\mathbf{2}$",
    "True logits: $0,\\;0$ (unchanged)\nQuest bound: $\\mathbf{0}$",
]
for cx,caption in zip(centers,captions):
    fig.text(cx,.11,caption,ha="center",va="center",fontsize=11,linespacing=1.65)
fig.text(.5,.955,"RoPE moves keys and Quest depends on the coordinate axes",ha="center",va="top",fontsize=15,weight="normal")
fig.savefig(OUT/"rope_geometry_tutorial.png",dpi=200,facecolor="white")
fig.savefig(OUT/"rope_geometry_tutorial.svg",facecolor="white")
print("Saved rope_geometry_tutorial.png and .svg; verified b/c logits and bounds.")
