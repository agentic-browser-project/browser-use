"""Figures for simulator/report_bsa_project.zh.md (BrowserSparseAttention project report).

All numbers are transcribed from the project's measured results:
  simulator/scoring_report.md, simulator/report_codesign.md, simulator/mixmax_design_report.md,
  simulator/report.md (Aug 7 speed study), simulator/CHUNKED_*.md (real chunk boundaries),
  simulator/runs/sparse3way-20260721/scoring_case_study/analysis/sweep_{v3,wn,explore}.md,
  BrowserSparseAttention README (branch shiqihe/mixmax-main / region-aware).
Run:  python simulator/figures/bsa_report/make_figures.py
Writes PNG + SVG next to this file.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_FIXED = "#4C72B0"      # fixed-16 pages (Quest / Block baselines)
C_TREE = "#DD8452"       # single-parameter semantic chunks (method 1)
C_REGION = "#55A868"     # region-aware chunks (method 2)
C_DENSE = "#333333"      # full attention
C_BUG = "#C44E52"        # legacy / bug path
C_GRAY = "#999999"
C_CODESIGN = "#8172B2"   # co-design (method 4)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------
# fig01: one decode step, full vs sparse attention
# ----------------------------------------------------------------------------
def fig01():
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2))
    n_tok = 28
    tok_w, tok_h = 0.32, 0.5
    x0 = 1.6

    def draw_tokens(ax, y, selected=None, alpha_unsel=0.25):
        for i in range(n_tok):
            sel = True if selected is None else (i in selected)
            ax.add_patch(Rectangle((x0 + i * tok_w, y), tok_w * 0.9, tok_h,
                                   facecolor=C_FIXED if sel else "#dddddd",
                                   edgecolor="white", alpha=1.0 if sel else 0.9))
        ax.text(x0 + n_tok * tok_w / 2, y - 0.28, "KV cache: one (k, v) pair per prompt token, per layer",
                ha="center", va="top", fontsize=9, color="#444")

    # ---- (a) full attention
    ax = axes[0]
    ax.set_xlim(0, 11.2)
    ax.set_ylim(-0.9, 1.95)
    ax.axis("off")
    ax.text(0.0, 1.9, "(a) Full attention: the query of the token being generated reads EVERY cached key/value",
            fontsize=10.5, weight="bold", va="top")
    ax.add_patch(Rectangle((0.2, 0.55), 0.9, 0.5, facecolor="#F4C7A1", edgecolor="#8a4b1f"))
    ax.text(0.65, 0.8, "q", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.65, 0.35, "step t,\nlayer l", ha="center", va="top", fontsize=8, color="#444")
    draw_tokens(ax, 0.55)
    for i in range(0, n_tok, 3):
        ax.add_patch(FancyArrowPatch((1.1, 0.8), (x0 + i * tok_w + 0.12, 1.08),
                                     arrowstyle="-|>", mutation_scale=7, color="#8a4b1f", alpha=0.55, lw=0.8))
    ax.text(x0, 1.55, "cost per generated token = read all N tokens x 48 layers   (N = 6k-30k for a browser page)",
            fontsize=9, color="#8a4b1f")

    # ---- (b) sparse attention
    ax = axes[1]
    ax.set_xlim(0, 11.2)
    ax.set_ylim(-0.9, 2.6)
    ax.axis("off")
    ax.text(0.0, 2.5, "(b) Sparse attention (query-aware KV selection): score cheap per-chunk summaries, read only the top chunks",
            fontsize=10.5, weight="bold", va="top")
    chunks = [(0, 3), (3, 8), (8, 12), (12, 18), (18, 21), (21, 25), (25, 28)]
    scores = [None, 1.4, 3.9, 0.7, 2.1, 5.2, None]  # None = always-included (sink / recent)
    order = sorted([i for i, s in enumerate(scores) if s is not None], key=lambda i: -scores[i])
    top = set(order[:2])
    selected = set()
    for ci, (a, b) in enumerate(chunks):
        if ci in top or scores[ci] is None:
            selected.update(range(a, b))
    ax.add_patch(Rectangle((0.2, 0.55), 0.9, 0.5, facecolor="#F4C7A1", edgecolor="#8a4b1f"))
    ax.text(0.65, 0.8, "q", ha="center", va="center", fontsize=12, weight="bold")
    draw_tokens(ax, 0.55, selected=selected)
    # chunk brackets + summary boxes
    for ci, (a, b) in enumerate(chunks):
        xa = x0 + a * tok_w
        xb = x0 + b * tok_w - tok_w * 0.1
        ax.plot([xa, xa, xb, xb], [1.12, 1.2, 1.2, 1.12], color="#555", lw=0.8)
        cx = (xa + xb) / 2
        if scores[ci] is None:
            ax.text(cx, 1.28, "always\nincluded", ha="center", va="bottom", fontsize=7.5, color="#444")
            continue
        col = "#F4C7A1" if ci in top else "#eeeeee"
        ax.add_patch(Rectangle((cx - 0.22, 1.32), 0.44, 0.34, facecolor=col, edgecolor="#8a4b1f" if ci in top else "#999"))
        ax.text(cx, 1.49, "M,m", ha="center", va="center", fontsize=7.5)
        ax.text(cx, 1.72, f"s={scores[ci]}", ha="center", va="bottom", fontsize=8,
                weight="bold" if ci in top else "normal", color="#8a4b1f" if ci in top else "#666")
        ax.add_patch(FancyArrowPatch((1.1, 0.95), (cx - 0.24, 1.45),
                                     arrowstyle="-|>", mutation_scale=7, color="#8a4b1f", alpha=0.5, lw=0.8))
    ax.text(x0, 2.12, "prefill (once per request): per chunk, per layer, store a summary of its keys (min/max envelope M,m or centroid)",
            fontsize=8.5, color="#444")
    ax.text(x0, -0.62, "decode (every generated token, every layer): score all chunk summaries with q  ->  keep top-k / top-B tokens "
                       "(+ sink & recent)  ->  attention over the selected pages only",
            fontsize=8.5, color="#444")
    ax.text(0.65, 0.35, "step t,\nlayer l", ha="center", va="top", fontsize=8, color="#444")
    save(fig, "fig01_decode_step")


# ----------------------------------------------------------------------------
# fig02: prompt anatomy + attention mass by region
# ----------------------------------------------------------------------------
def fig02():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    rows = [("short step\n(~8k tokens)", [5300, 500, 2500, 100]),
            ("long episode\n(~20k tokens)", [5300, 10000, 5000, 100])]
    labels = ["system prompt (rules, output format, examples)", "task + history of previous steps",
              "DOM observation (one element per line)", "URL / tail"]
    cols = ["#9ecae1", "#c6dbef", C_TREE, "#bbbbbb"]
    for r, (name, parts) in enumerate(rows):
        left = 0
        for p, lab, c in zip(parts, labels, cols):
            ax.barh(r, p, left=left, color=c, edgecolor="white", label=lab if r == 0 else None)
            if p > 800:
                ax.text(left + p / 2, r, f"{p/1000:.1f}k", ha="center", va="center", fontsize=8.5)
            left += p
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("tokens")
    ax.set_xlim(0, 21500)
    ax.legend(loc="upper center", frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, -0.28), ncol=2)
    ax.set_title("(a) What a browser-agent prompt contains", loc="left")
    ax.annotate("action targets (needles)\nlive only here", xy=(5300 + 10000 + 2500, 1.2), xytext=(9500, 1.55),
                fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color="#8a4b1f"), color="#8a4b1f")
    ax.set_ylim(-0.6, 1.9)

    ax = axes[1]
    regions = ["sink\n(first 4 tok)", "system\nprompt", "task +\nhistory", "DOM", "other"]
    mass = [54, 22, 8, 12, 4]
    colors = ["#777777", "#9ecae1", "#c6dbef", C_TREE, "#dddddd"]
    ax.bar(range(5), mass, color=colors, edgecolor="white")
    for i, m in enumerate(mass):
        ax.text(i, m + 1, f"{m}%", ha="center", fontsize=9)
    ax.set_xticks(range(5))
    ax.set_xticklabels(regions, fontsize=8, rotation=0)
    ax.set_ylabel("share of full-attention mass (%)")
    ax.set_ylim(0, 65)
    ax.set_title("(b) Where full attention puts its weight\n(15 real steps, all layers; region_sweep.py)", loc="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig02_prompt_anatomy")


# ----------------------------------------------------------------------------
# fig03: real chunk boundaries on one Allrecipes step (tokens 9584-9866)
# ----------------------------------------------------------------------------
def fig03():
    lo, hi = 9584, 9866
    rows = [
        ("fixed-16 pages\n(Quest / Block)", [(s, min(s + 15, hi)) for s in range(9584, hi + 1, 16)], C_FIXED),
        ("old semantic leaf chunks\n(one rule, merge>=16 / split<=256)",
         [(9524, 9650), (9651, 9652), (9653, 9689), (9690, 9691), (9692, 9756), (9757, 9758), (9759, 9849), (9850, 9866)], C_TREE),
        ("main: subtree chunks\n(merge>=16 / split<=32)",
         [(9588, 9619), (9620, 9650), (9651, 9652), (9653, 9684), (9685, 9691), (9692, 9723), (9724, 9755),
          (9756, 9758), (9759, 9790), (9791, 9822), (9823, 9849), (9850, 9865)], "#8c8c8c"),
        ("region-aware (this work)\nDOM: merge>=6 / split<=64",
         [(9588, 9650), (9651, 9691), (9692, 9755), (9756, 9758), (9759, 9822), (9823, 9849), (9850, 9865)], C_REGION),
        ("content-driven prototype (V2)\ncap 32, line/bracket aligned",
         [(9573, 9594), (9595, 9606), (9607, 9624), (9625, 9650), (9651, 9675), (9676, 9689), (9690, 9699),
          (9700, 9731), (9732, 9756), (9757, 9788), (9789, 9795), (9796, 9810), (9811, 9819), (9820, 9849), (9850, 9863)], "#c9a227"),
    ]
    elements = [  # (start, end, label)  from the V2 dump's element-aligned spans
        (9607, 9624, "[699]<header>"), (9625, 9650, "[700]<a skip>"), (9651, 9675, "[727]<a logo>"),
        (9676, 9689, "[729]<div>"), (9700, 9731, "[752]<li> [797]<div>"), (9732, 9756, "[61]<form search>"),
        (9757, 9795, "[58]<input search box>"), (9796, 9810, "[113]<button>"), (9811, 9819, "[814]<div>"),
        (9820, 9849, "[114]<a Newsletters>"), (9850, 9863, "[835]<a Sweepstakes>"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.0))
    h = 0.62
    for r, (name, segs, col) in enumerate(rows):
        y = len(rows) - 1 - r
        for j, (a, b) in enumerate(segs):
            a2, b2 = max(a, lo), min(b, hi)
            if b2 < a2:
                continue
            shade = 0.9 if j % 2 == 0 else 0.55
            ax.add_patch(Rectangle((a2, y - h / 2), b2 - a2 + 1, h, facecolor=col, alpha=shade, edgecolor="white", lw=1.2))
            n = b - a + 1
            if b2 - a2 + 1 >= 14:
                ax.text((a2 + b2 + 1) / 2, y, str(n), ha="center", va="center", fontsize=7.5, color="white", weight="bold")
    # needle span
    ax.axvspan(9757, 9796, color="#C44E52", alpha=0.13, lw=0)
    ax.text(9776.5, len(rows) - 0.25, "needle: [58]<input> (the search box)", ha="center", va="bottom", fontsize=8.5, color="#C44E52")
    # element labels
    for k, (a, b, lab) in enumerate(elements):
        yy = -0.85 - 0.26 * (k % 4)
        ax.plot([a, b + 1], [yy + 0.08, yy + 0.08], color="#444", lw=1)
        ax.text((a + b + 1) / 2, yy, lab, ha="center", va="top", fontsize=6.8, color="#333")
    ax.set_xlim(lo, hi + 1)
    ax.set_ylim(-2.0, len(rows) - 0.1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5)
    ax.set_xlabel("token position in the prompt (Allrecipes home page, step 2 of a real trajectory; numbers = chunk length in tokens)")
    for s in ["left", "top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Five ways to cut the same DOM region into selection units (tokens 9584-9866 of an 11,062-token prompt)", loc="left")
    save(fig, "fig03_real_chunking")


# ----------------------------------------------------------------------------
# fig04: toy scoring example (3 chunks x 5 scorers)
# ----------------------------------------------------------------------------
def fig04():
    chunks = ["A: needle chunk\n(1 strong key + 3 unrelated)", "B: large mixed chunk\n(no matching key, wide box)",
              "C: small homogeneous\n(4 identical medium keys)"]
    scorers = ["true best q.k in chunk", "Quest upper bound\n(per-dim max)", "TSA original 2-corner\nmax(q.M, q.m)",
               "centroid q.mean", "mixmax_wn (alpha=.25)\nA,C=16 tok, B=256 tok"]
    vals = np.array([
        [10, 0, 6],     # truth
        [10, 18, 6],    # quest
        [2, 0, 6],      # 2-corner
        [1, 0, 6],      # centroid
        [10, 9, 6],     # mixmax_wn
    ])
    picks = [0, 1, 2, 2, 0]
    cols = [C_DENSE, C_FIXED, C_BUG, C_GRAY, C_REGION]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    w = 0.15
    x = np.arange(3)
    for i, (name, c) in enumerate(zip(scorers, cols)):
        xi = x + (i - 2) * w
        bars = ax.bar(xi, vals[i], w, color=c, label=name, edgecolor="white")
        for j, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.25, str(vals[i][j]), ha="center", fontsize=7.5)
            if j == picks[i] and i > 0:
                ax.text(b.get_x() + b.get_width() / 2, -1.6, "pick" if j == 0 else "pick x", ha="center", fontsize=7,
                        color=C_REGION if j == 0 else C_BUG, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(chunks, fontsize=8.5)
    ax.set_ylabel("chunk score (budget = 1 chunk)")
    ax.set_ylim(-2.4, 21)
    ax.axhline(0, color="#999", lw=0.6)
    ax.legend(frameon=False, fontsize=7.8, ncol=1, loc="upper right")
    ax.set_title("Toy example (D=4, q=[2,-2,1,-1]): full attention gives chunk A 93% of the mass; which scorer finds it?", loc="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig04_toy_scoring")


# ----------------------------------------------------------------------------
# fig05: head aggregation toy (from report_codesign 4.2)
# ----------------------------------------------------------------------------
def fig05():
    rows = ["C1 rules block (80 tok)", "C2 task block (30 tok)", "E1 banner <div> (12 tok)",
            "E3 search <input> (6 tok)  <- needle", "N1 nav <a> (6 tok)"]
    g = np.array([[1.4, 1.4, 1.4, 1.4], [1.5, 1.5, 1.5, 1.5], [2.6, 0.5, 0.4, 0.5],
                  [0.4, 0.4, 0.3, 3.2], [1.2, 1.2, 1.1, 1.1]])
    mean = g.mean(1)
    mx = g.max(1)
    data = np.concatenate([g, mean[:, None], mx[:, None]], axis=1)
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    im = ax.imshow(data, cmap="YlOrBr", vmin=0, vmax=3.4, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="black" if data[i, j] < 2.4 else "white")
    ax.set_xticks(range(6))
    ax.set_xticklabels(["group g1", "g2", "g3", "g4\n(retrieval head)", "head-MEAN\n(deployed)", "head-MAX"], fontsize=8.5)
    ax.set_yticks(range(5))
    ax.set_yticklabels(rows, fontsize=8.5)
    ax.axvline(3.5, color="white", lw=3)
    ax.set_title("Toy: per-KV-head-group scores of 5 chunks, then aggregated (4 groups = 32 query heads / 8)", loc="left", fontsize=9.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="score")
    save(fig, "fig05_head_aggregation_toy")


# ----------------------------------------------------------------------------
# fig06: measured envelope width growth vs chunk length
# ----------------------------------------------------------------------------
def fig06():
    L = np.array([8, 16, 32, 64, 128, 256])
    width = np.array([0.767, 1.000, 1.227, 1.449, 1.656, 1.858])
    corr = (16 / L) ** 0.25
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(L, width, "o-", color=C_TREE, label="measured width term, relative to L=16\n(real K dumps, 2 dumps x 3 requests x 4 layers)")
    ax.plot(L, width * corr, "s-", color=C_REGION, label="after x (16/L)^0.25 correction")
    ax.plot(L, (16 / L) ** -0.245, "--", color="#888", lw=1, label="power-law fit, slope beta = 0.245")
    ax.plot(L, np.sqrt(np.log(L) / np.log(16)), ":", color="#bbb", lw=1.2, label="iid-Gaussian extreme-value prediction sqrt(ln L / ln 16)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(L)
    ax.set_xticklabels(L)
    ax.set_xlabel("chunk length L (tokens)")
    ax.set_ylabel("width term  sum_d |q_d| (M-m)_d / 2   (relative)")
    ax.set_ylim(0.6, 2.0)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    ax.set_title("Why long chunks look optimistic: the envelope half-width grows with L", loc="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig06_width_growth")


# ----------------------------------------------------------------------------
# fig07: alpha sweep (sweep_wn.md, tree chunks, n=15 dumps)
# ----------------------------------------------------------------------------
def fig07():
    alphas = [0, 0.25, 0.5, 0.75, 1.0]
    k64 = [0.885, 0.749, 0.417, 0.356, 0.312]
    b4096 = [0.179, 0.805, 0.791, 0.749, 0.738]
    b2048 = [0.085, 0.641, 0.613, 0.584, 0.568]
    size = [107.9, 55.9, 27.0, 20.1, 16.9]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(alphas, k64, "o-", color=C_GRAY, label="budget = 64 chunks (chunk-count accounting)")
    ax.plot(alphas, b4096, "s-", color=C_TREE, label="budget = 4096 tokens")
    ax.plot(alphas, b2048, "^-", color=C_REGION, label="budget = 2048 tokens")
    ax.axvline(0.25, color="#ccc", lw=1, ls="--")
    ax.set_xlabel("width-normalization exponent alpha in (16/L)^alpha")
    ax.set_ylabel("needle hit rate (per layer, per decode step)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(a) needle hit vs alpha (tree chunks, 15 real steps)", loc="left", fontsize=9.5)
    ax = axes[1]
    ax.plot(alphas, size, "o-", color=C_TREE)
    ax.axhline(39, color="#999", ls=":", lw=1)
    ax.text(0.55, 41, "corpus mean chunk length = 39 tok", fontsize=8, color="#666")
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean length of selected chunks (tokens) at k=64")
    ax.set_ylim(0, 120)
    ax.set_title("(b) alpha=0 selects the fattest chunks (width spam)", loc="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig07_alpha_sweep")


# ----------------------------------------------------------------------------
# fig08: offline needle hit vs token budget, tree vs fixed16
# ----------------------------------------------------------------------------
def fig08():
    B = [1024, 2048, 4096, 8192, 16384]
    series = [
        ("fixed-16 + Quest bound (per-head max)", [0.259, 0.410, 0.646, 0.963, 1.000], C_FIXED, "o", "-"),
        ("fixed-16 + Block (centroid softmax, head-max)", [0.371, 0.555, 0.765, 0.958, 1.000], C_FIXED, "s", "--"),
        ("tree + Quest bound, no normalization", [0.039, 0.067, 0.133, 0.936, 1.000], C_TREE, "o", "-"),
        ("tree + centroid", [0.405, 0.578, 0.748, 0.967, 1.000], C_TREE, "^", ":"),
        ("tree + mixmax_wn alpha=.25 (this work)", [0.404, 0.641, 0.805, 0.969, 0.999], C_REGION, "D", "-"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, y, c, m, ls in series:
        ax.plot(B, y, marker=m, ls=ls, color=c, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(B)
    ax.set_xticklabels([f"{b//1024}k" for b in B])
    ax.set_xlabel("KV token budget per layer per decode step")
    ax.set_ylabel("needle hit rate (offline, 15 real steps, all layers)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right")
    ax.set_title("Offline: the un-normalized bound collapses on variable-length chunks at token budgets", loc="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig08_offline_budget_curves")


# ----------------------------------------------------------------------------
# fig09: end-to-end agree at B4096 (half1, 190 index steps)
# ----------------------------------------------------------------------------
def fig09():
    names = ["full attention (dense)", "fixed-16 + Block", "fixed-16 + Quest", "fixed-16 + mixmax",
             "tree (merge>=16) + mixmax_wn + token budget  [method 1]", "tree + mixmax, top-64 chunks",
             "tree + Quest bound, top-64 chunks", "tree + original TSA path (KV layout bug)"]
    vals = [97, 99, 98, 96, 96, 91, 89, 55]
    cols = [C_DENSE, C_FIXED, C_FIXED, C_FIXED, C_REGION, C_TREE, C_TREE, C_BUG]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, edgecolor="white")
    for yi, v in zip(y, vals):
        ax.text(v + 1, yi, f"{v}  ({v/190*100:.1f}%)", va="center", fontsize=8.5)
    ax.axvspan(97 - 8, 97 + 8, color="#000", alpha=0.06, lw=0)
    ax.axvline(97, color=C_DENSE, ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("agree with the reference action (out of 190 index steps, teacher-forced replay)")
    ax.set_xlim(0, 125)
    ax.set_title("End-to-end at budget 4096 tokens: every faithful scorer sits in the dense noise band\n"
                 "(dashed = full attention 97; grey band = +-8 steps, the bootstrap 95% width of a paired difference)",
                 loc="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig09_e2e_B4096")


# ----------------------------------------------------------------------------
# fig10: online agree vs budget (strat20, 132 index steps)
# ----------------------------------------------------------------------------
def fig10():
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.axhline(57, color=C_DENSE, ls="--", lw=1.2, label="full attention (57)")
    ax.plot([1024, 2048, 4096], [55, 55, 59], "o-", color=C_FIXED, label="fixed-16 + Quest")
    ax.plot([2048, 3072, 4096], [55, 53, 60], "s--", color=C_FIXED, label="fixed-16 + Block")
    ax.plot([2048, 3072, 4096], [48, 57, 58], "^-", color=C_TREE, label="method 1: one-rule semantic chunks + mixmax_wn")
    ax.plot([1024, 2048], [53, 57], "D-", color=C_REGION, lw=2.2, ms=8, label="method 2: region-aware chunks + sys floor 0.25")
    ax.plot([1024, 2048], [46, 52], "v:", color=C_CODESIGN, label="method 4: co-design (rejected; B2048 has 17 timeouts)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 2048, 3072, 4096])
    ax.set_xticklabels(["1k", "2k", "3k", "4k"])
    ax.set_xlabel("KV token budget per layer per decode step")
    ax.set_ylabel("agree (out of 132 index steps)")
    ax.set_ylim(40, 64)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right")
    ax.set_title("Online replay vs budget (strat20: 20 tasks / 15 sites / 132 index steps)", loc="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig10_budget_online")


# ----------------------------------------------------------------------------
# fig11: method 1 -> method 2 at B2048 on two evaluation sets
# ----------------------------------------------------------------------------
def fig11():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    sets = [("strat20 (132 index steps)", [48, 57, 55, 57], 132),
            ("half1 (190 index steps)", [82, 94, 91, 97], 190)]
    labels = ["method 1\n(tree, one rule)", "method 2\n(region-aware)", "fixed-16\nQuest", "full\nattention"]
    cols = [C_TREE, C_REGION, C_FIXED, C_DENSE]
    for ax, (name, vals, n) in zip(axes, sets):
        bars = ax.bar(range(4), vals, color=cols, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v} ({v/n*100:.0f}%)", ha="center", fontsize=8.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(min(vals) - 12, max(vals) + 9)
        ax.set_ylabel("agree")
        ax.set_title(f"budget 2048 tokens, {name}", loc="left", fontsize=9.5)
        ax.text(1, vals[1] + 3.6, f"+{vals[1]-vals[0]} vs method 1", ha="center", color=C_REGION, weight="bold", fontsize=9)
    fig.suptitle("Same chunk tree, same scorer, same budget: only the per-region chunk sizes and the system-prompt floor changed", fontsize=9.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig11_method_evolution")


# ----------------------------------------------------------------------------
# fig12: the two tight-budget failure mechanisms (B2048 dumps)
# ----------------------------------------------------------------------------
def fig12():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    size = [138, 139, 109, 142, 219, 178, 50, 28, 27, 136]
    tree = [0.01, 0.10, 0.09, 0.08, 0.20, 0.20, 0.46, 0.65, 0.87, 0.50]
    block = [0.24, 0.35, 0.15, 0.29, 0.66, 0.54, 0.52, 0.29, 0.79, 0.33]
    ax.scatter(size, tree, color=C_TREE, s=55, label="tree chunk actually selected (method 1, B2048)", zorder=3)
    ax.scatter(size, block, color=C_FIXED, s=55, marker="s", label="same needle, fixed-16 Block pages (offline, same queries)", zorder=3)
    for s, t, b in zip(size, tree, block):
        ax.plot([s, s], [t, b], color="#bbb", lw=0.8, zorder=1)
    ax.axvspan(100, 230, color=C_TREE, alpha=0.07, lw=0)
    ax.text(165, 0.95, "needle inside a large <form>/<div> subtree\n(search inputs, combobox)", ha="center", fontsize=8, color="#8a4b1f")
    ax.set_xlabel("size of the chunk containing the needle (tokens)")
    ax.set_ylabel("needle read (fraction of decode steps)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    ax.set_title("(a) Mechanism A: big subtrees starve their needle", loc="left", fontsize=9.5)

    ax = axes[1]
    names = ["method 1\nB2048", "method 1\nB4096", "fixed-16 Quest\nB2048 (offline)"]
    lo = [7, 36, 27]
    hi = [14, 62, 27]
    mid = [(a + b) / 2 for a, b in zip(lo, hi)]
    ax.bar(range(3), mid, color=[C_TREE, C_TREE, C_FIXED], edgecolor="white", alpha=0.9)
    ax.errorbar(range(3), mid, yerr=[[m - a for m, a in zip(mid, lo)], [b - m for m, b in zip(mid, hi)]],
                fmt="none", ecolor="#333", capsize=5)
    for i, (a, b) in enumerate(zip(lo, hi)):
        ax.text(i, b + 2.5, f"{a}-{b}%" if a != b else f"{a}%", ha="center", fontsize=8.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("system-prompt tokens covered (%)")
    ax.set_ylim(0, 75)
    ax.set_title("(b) Mechanism B: instruction coverage collapses", loc="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig12_mechanisms_B2048")


# ----------------------------------------------------------------------------
# fig13: ablation at B2048 (strat20): chunking vs floor
# ----------------------------------------------------------------------------
def fig13():
    names = ["fixed-16\nQuest", "fixed-16\nBlock", "method 1\n(one rule)", "method 1\n+ floor .05",
             "method 2\n(region\n+ floor .25)", "full\nattention"]
    agree = [55, 55, 48, 50, 57, 57]
    none = [31, 25, 40, 26, 19, 33]
    cols = [C_FIXED, C_FIXED, C_TREE, C_TREE, C_REGION, C_DENSE]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, vals, lab, lim in zip(axes, [agree, none], ["agree (higher is better)", "'none' outputs (lower is better)"], [(40, 62), (0, 46)]):
        bars = ax.bar(range(6), vals, color=cols, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4, str(v), ha="center", fontsize=8.5)
        ax.set_xticks(range(6))
        ax.set_xticklabels(names, fontsize=7.8)
        ax.set_ylabel(lab)
        ax.set_ylim(*lim)
    axes[0].set_title("(a) budget 2048 tokens, strat20 132 index steps", loc="left", fontsize=9.5)
    axes[1].set_title("(b) the floor alone fixes 'none'; element chunks fix agree", loc="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig13_ablation_floor_chunking")


# ----------------------------------------------------------------------------
# fig14: number of scored units
# ----------------------------------------------------------------------------
def fig14():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    names = ["fixed-16\npages", "V2\nprototype\n(cap 32)", "main\nsubtree\n(16/32)", "region-\naware\n(this work)", "old\nsemantic\n(16/256)", "old\nregion\n(leaf-based)"]
    vals = [692, 446, 368, 204, 185, 102]
    cols = [C_FIXED, "#c9a227", "#8c8c8c", C_REGION, C_TREE, C_REGION]
    bars = ax.bar(range(6), vals, color=cols, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, str(v), ha="center", fontsize=8.5)
    ax.set_xticks(range(6))
    ax.set_xticklabels(names, fontsize=7.8)
    ax.set_ylabel("selection units for one 11,062-token prompt")
    ax.set_title("(a) Same prompt, six chunkers", loc="left", fontsize=9.5)
    ax = axes[1]
    x = np.arange(2)
    w = 0.26
    ax.bar(x - w, [424, 1672], w, color=C_FIXED, label="fixed-16 pages")
    ax.bar(x, [78, 400], w, color=C_TREE, label="tree + mixmax_wn (method 1)")
    ax.bar(x + w, [53, 239], w, color=C_REGION, label="region-aware (method 2)")
    for xi, vs in zip([x - w, x, x + w], [[424, 1672], [78, 400], [53, 239]]):
        for a, v in zip(xi, vs):
            ax.text(a, v + 20, str(v), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["6.2k-token prompt", "23k-token prompt"])
    ax.set_ylabel("units scored per layer per decode step")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(b) Units scored in the selection micro-benchmark", loc="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig14_chunk_counts")


# ----------------------------------------------------------------------------
# fig15: selection latency
# ----------------------------------------------------------------------------
def fig15():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    names = ["region-aware\n(method 2)", "tree + mixmax_wn\n(method 1)", "fixed-16\ncentroid", "fixed-16\nenvelope"]
    v6 = [51.7, 53.7, 65.4, 74.1]
    v23 = [65.0, 73.7, 96.9, 114.0]
    cols = [C_REGION, C_TREE, C_FIXED, C_FIXED]
    x = np.arange(4)
    w = 0.36
    b1 = ax.bar(x - w / 2, v6, w, color=cols, edgecolor="white", alpha=0.75, label="6.2k-token prompt")
    b2 = ax.bar(x + w / 2, v23, w, color=cols, edgecolor="white", label="23k-token prompt", hatch="//")
    for bb, vv in zip(list(b1) + list(b2), v6 + v23):
        ax.text(bb.get_x() + bb.get_width() / 2, vv + 1.5, f"{vv:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("select time per layer per step (us)")
    from matplotlib.patches import Patch
    ax.legend(frameon=False, fontsize=8, handles=[Patch(facecolor="#888", alpha=0.75, label="6.2k-token prompt"),
                                                  Patch(facecolor="#888", hatch="//", label="23k-token prompt")])
    ax.set_title("(a) CUDA selector on GB10: fewer units -> 1.4-1.75x cheaper selection", loc="left", fontsize=9.5)
    ax = axes[1]
    names = ["CUDA kernel", "PyTorch path", "PyTorch +\nreselect every 8"]
    v6 = [53.7, 2614, 1315]
    v23 = [73.7, 2064, 427]
    x = np.arange(3)
    ax.bar(x - w / 2, v6, w, color=C_TREE, alpha=0.75)
    ax.bar(x + w / 2, v23, w, color=C_TREE, hatch="//")
    for xi, vv in zip(list(x - w / 2) + list(x + w / 2), v6 + v23):
        ax.text(xi, vv * 1.15, f"{vv:.0f}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("us per layer per step (log)")
    ax.set_title("(b) tree + mixmax_wn: kernel vs reference path", loc="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig15_select_latency")


# ----------------------------------------------------------------------------
# fig16: where decode time goes; when sparsity pays
# ----------------------------------------------------------------------------
def fig16():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    parts = [("MoE experts (grouped_mm)", 77.1, "#c44e52"), ("Python loop, rope, residual, decode/argmax", 42.0, "#bbbbbb"),
             ("q/k/v/o proj + norms + router", 13.1, "#8c8c8c"), ("paged attention over full 29.5k KV (roofline lower bound)", 13.0, C_FIXED),
             ("lm_head", 3.4, "#dddddd"), ("sparse page selection (x48 layers)", 2.5, C_REGION)]
    left = 0
    for name, v, c in parts:
        ax.barh(0, v, left=left, color=c, edgecolor="white", label=f"{name}: {v:.1f} ms")
        left += v
    ax.set_xlim(0, 155)
    ax.set_yticks([])
    ax.set_xlabel("ms per decode step (batch 1, Qwen3-VL-30B-A3B, 29.5k-token prompt, GB10; measured Aug 2026)")
    ax.legend(frameon=False, fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax.set_title("(a) Attention is <=15% of a decode step on this model/machine:\nAmdahl ceiling of perfect sparsity = 1.09x (1.43x on an ideal machine)", loc="left", fontsize=9.2)
    ax = axes[1]
    bs = [1, 2, 4, 8, 16, 32, 64]
    tsa = [7.91, 8.50, 8.97, 10.32, 11.33, 13.06, 18.72]
    sgl = [13.8, 15.3, 16.8, 19.8, 26.0, 37.8, 66.0]
    ax.plot(bs, sgl, "o-", color=C_DENSE, label="SGLang full attention")
    ax.plot(bs, tsa, "s-", color=C_REGION, label="TreeSparse + CUDA graph (top-128 chunks)")
    for b, t, s in zip(bs, tsa, sgl):
        ax.text(b, s + 2, f"{s/t:.2f}x", ha="center", fontsize=7.5, color="#444")
    ax.set_xscale("log", base=2)
    ax.set_xticks(bs)
    ax.set_xticklabels(bs)
    ax.set_xlabel("batch size")
    ax.set_ylabel("time per output token (ms)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(b) Upstream TSA measurement (Qwen3-VL-8B dense, B200,\n9.6k-token prompt): sparsity pays when attention dominates", loc="left", fontsize=9.2)
    fig.tight_layout()
    save(fig, "fig16_speed_arithmetic")


# ----------------------------------------------------------------------------
# fig17: co-design: offline proxy vs online outcome
# ----------------------------------------------------------------------------
def fig17():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    ax = axes[0]
    names = ["fixed-16\nBlock", "method 2", "+ 3-way\npartition", "+ interactive\nfirst", "+ DOM head-max\n= method 4"]
    b1 = [0.331, 0.296, 0.302, 0.399, 0.465]
    b2 = [0.444, 0.515, 0.537, 0.679, 0.695]
    x = np.arange(5)
    w = 0.36
    ax.bar(x - w / 2, b1, w, color="#bbbbbb", label="B1024")
    ax.bar(x + w / 2, b2, w, color=C_CODESIGN, label="B2048")
    for xi, v in zip(list(x - w / 2) + list(x + w / 2), b1 + b2):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("offline needle hit (23 failure-enriched dumps)")
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(a) Offline proxy predicted a big win for method 4", loc="left", fontsize=9.5)
    ax = axes[1]
    names = ["fixed-16 Quest", "method 2", "method 4"]
    b1 = [55, 53, 46]
    b2 = [55, 57, 52]
    x = np.arange(3)
    ax.bar(x - w / 2, b1, w, color="#bbbbbb", label="B1024 (n=132 / 128)")
    ax.bar(x + w / 2, b2, w, color=C_CODESIGN, label="B2048 (n=132; method 4: 17 timeouts)")
    for xi, v in zip(list(x - w / 2) + list(x + w / 2), b1 + b2):
        ax.text(xi, v + 0.5, str(v), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("online agree (strat20)")
    ax.set_ylim(35, 62)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("(b) Online replay rejected it", loc="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig17_codesign_offline_vs_online")


if __name__ == "__main__":
    for f in [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11, fig12, fig13, fig14, fig15, fig16, fig17]:
        f()
