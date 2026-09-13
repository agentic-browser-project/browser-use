"""Figures for simulator/report_bsa_project_simple_sep_10.md (progress Sep 3-10, 2026).

Numbers transcribed from simulator/report_anchor.md (anchor / exponent study, verification section),
report_codesign.md section 8 (sglang validation) and the verification chain logs on spark00.
Run:  python simulator/figures/bsa_report/make_figures_sep10.py   (writes fig18..fig23 PNG+SVG here)
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
                     "figure.dpi": 110, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False})
C_FIXED, C_TREE, C_REGION, C_DENSE, C_BUG, C_GRAY, C_CODESIGN = "#4C72B0", "#DD8452", "#55A868", "#333333", "#C44E52", "#999999", "#8172B2"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches="tight")
    plt.close(fig); print("wrote", name)


# fig18: serving speed, standalone server vs sglang (same method, same 132 steps)
def fig18():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axes[0]
    ax.bar([0, 1], [20700, 967], color=[C_GRAY, C_REGION], width=0.6)
    ax.set_yscale("log"); ax.set_xticks([0, 1]); ax.set_xticklabels(["standalone\nserve.py", "sglang\nbsa_v2"])
    ax.set_ylabel("seconds per 132-step evaluation (log)"); ax.set_title("(a) wall-clock, B = 2048")
    for x, v, t in ((0, 20700, "5.75 h"), (1, 967, "16 min")):
        ax.text(x, v * 1.25, t, ha="center")
    ax = axes[1]
    ax.bar([0, 1], [157, 7.3], color=[C_GRAY, C_REGION], width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["standalone", "sglang"]); ax.set_ylabel("seconds per step"); ax.set_title("(b) per agent step")
    for x, v in ((0, 157), (1, 7.3)):
        ax.text(x, v + 4, f"{v:g}", ha="center")
    ax = axes[2]
    ax.bar([0, 1], [12, 90], color=[C_GRAY, C_REGION], width=0.6)
    ax.errorbar([1], [90], yerr=[[7], [6]], fmt="none", ecolor="k", capsize=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["standalone", "sglang"]); ax.set_ylabel("GPU utilisation (%)"); ax.set_ylim(0, 100)
    ax.set_title("(c) GPU busy while evaluating"); ax.text(1, 97, "83–96%", ha="center")
    fig.suptitle("Same method (region-aware chunks, mixmax_wn, sys floor 0.25) on the same GB10, 132 real steps", y=1.03)
    save(fig, "fig18_sglang_speed")


# fig19: cross-implementation equivalence + accuracy
def fig19():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    layers = ["layer 0", "layer 16", "layer 32"]; jac = [0.933, 0.950, 0.769]; smaller = [0.969, 0.985, 0.873]
    x = np.arange(3)
    ax.bar(x - 0.18, jac, 0.36, color=C_REGION, label="token-level Jaccard")
    ax.bar(x + 0.18, smaller, 0.36, color=C_TREE, label="overlap / smaller set")
    ax.set_xticks(x); ax.set_xticklabels(layers); ax.set_ylim(0, 1.05); ax.set_ylabel("selected-token overlap")
    ax.set_title("(a) standalone selector vs sglang backend, same prompt"); ax.legend(loc="lower left")
    ax.text(1, 1.0, "chunk boundaries: 149 / 149 identical", ha="center", fontsize=9)
    ax = axes[1]
    names = ["full attention\n(sglang)", "sparse, standalone\n(region+subtree)", "sparse, sglang\n(same config)"]
    vals = [57, 57, 59]; cols = [C_DENSE, C_GRAY, C_REGION]
    ax.bar(range(3), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, str(v), ha="center")
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=8.5); ax.set_ylim(0, 70); ax.set_ylabel("agree (of 132 steps)")
    ax.set_title("(b) accuracy, budget 2048")
    save(fig, "fig19_sglang_equivalence")


# fig20: what the length correction corrects
def fig20():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    L = np.array([5, 12, 24, 48, 96, 192])
    c = [-0.51, -0.49, -0.36, -0.38, 0.18, -0.54]; w = [8.33, 12.17, 14.74, 18.10, 20.01, 24.12]
    bound = [7.82, 11.67, 14.38, 17.71, 20.19, 23.58]; tmax = [1.01, 1.88, 2.53, 3.15, 3.83, 3.82]
    ax.plot(L, bound, "o-", color=C_BUG, label="Quest bound  c + w")
    ax.plot(L, w, "s--", color=C_TREE, label="width term  w")
    ax.plot(L, tmax, "^-", color=C_DENSE, label="true max  max_i q·k_i")
    ax.plot(L, c, "v-", color=C_REGION, label="center term  c")
    ax.fill_between(L, tmax, bound, color=C_BUG, alpha=0.12, lw=0, label="Quest overestimate (bound − true max)")
    ax.set_xscale("log", base=2); ax.set_xticks(L); ax.set_xticklabels(["1–8", "9–16", "17–32", "33–64", "65–128", "129–256"])
    ax.set_xlabel("chunk length L (tokens, log scale)"); ax.set_ylabel("logit units"); ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Per-chunk score terms vs chunk length (23 real prompts, 4 layers)")
    ax.annotate("overestimate\n6.8 logits", xy=(5, 4.4), ha="center", fontsize=8.5, color=C_BUG)
    ax.annotate("overestimate\n19.8 logits", xy=(192, 13.5), ha="center", fontsize=8.5, color=C_BUG)
    save(fig, "fig20_bound_bias")


# fig21: where the growth of the width term comes from (RoPE bands)
def fig21():
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    bands = ["0–7\n(θ≥0.19)", "8–15", "16–23", "24–31", "32–39", "40–47", "48–55", "56–63\n(θ≤1.4e-6)"]
    ex = [0.199, 0.328, 0.289, 0.214, 0.216, 0.223, 0.244, 0.260]
    cols = [C_GRAY, C_BUG, C_BUG, C_TREE, C_TREE, C_TREE, C_TREE, C_TREE]
    ax.bar(range(8), ex, color=cols, width=0.65)
    ax.axhline(0.248, color=C_DENSE, ls="--", lw=1); ax.text(7.45, 0.252, "all dims: 0.248", ha="right", fontsize=8.5)
    ax.axhline(0.170, color=C_GRAY, ls=":", lw=1); ax.text(7.45, 0.174, "iid Gaussian keys (extreme value): 0.17", ha="right", fontsize=8.5, color="#555555")
    ax.set_xticks(range(8)); ax.set_xticklabels(bands, fontsize=8.5); ax.set_ylim(0, 0.37)
    ax.set_xlabel("RoPE dimension pair j  (rotation angle per token θ_j = (5·10⁶)^(−2j/128))"); ax.set_ylabel("growth exponent of key range, L = 8 → 256")
    ax.set_title("Envelope width growth by RoPE frequency band (Qwen3-VL-30B-A3B keys, 12 prompts × 4 layers)")
    ax.annotate("angle crosses ~1 rad\nwithin 8–256 tokens", xy=(1.5, 0.33), xytext=(3.2, 0.33), fontsize=8.5, color=C_BUG, va="center",
                arrowprops=dict(arrowstyle="->", color=C_BUG))
    save(fig, "fig21_rope_bands")


# fig22: online scans of alpha and A
def fig22():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    alphas = [0.125, 0.25, 0.375, 0.5]
    s20 = np.array([45, 54, 51, 36]) / 132 * 100; h2 = np.array([77, 90, 70, 73]) / 179 * 100
    ax.plot(alphas, s20, "o-", color=C_TREE, label="strat20 (132 steps)")
    ax.plot(alphas, h2, "s-", color=C_REGION, label="half2 (179 new steps, other tasks)")
    ax.axvspan(0.192, 0.279, color=C_GRAY, alpha=0.18, lw=0); ax.text(0.236, 22.5, "measured β\n0.19–0.28", ha="center", fontsize=8, color="#555555")
    ax.set_xticks(alphas); ax.set_xlabel("exponent α  (A = 16, budget 1024)"); ax.set_ylabel("agree (%)"); ax.set_ylim(20, 56)
    ax.set_title("(a) α: the peak is at 0.25 on both task sets"); ax.legend(loc="upper right", fontsize=8.5)
    ax = axes[1]
    xs20 = [0.35, 1, 4, 8, 16, 32, 47.6, 64, 79.3]; d20 = [-3, -2, -1, 0, 0, -1, -6, -8, -2]        # strat20, 132 steps, vs A=16 (54)
    xh2 = [1, 4, 8, 16, 64]; dh2 = [3, -1, 2, 0, 3]                                              # half2, 179 steps, vs A=16 (90)
    ax.axhspan(-3, 3, color=C_GRAY, alpha=0.18, lw=0); ax.text(0.5, 3.4, "request-stream noise (±3 per 132 steps)", fontsize=8, color="#555555")
    ax.plot(xs20, d20, "o-", color=C_TREE, label="strat20 (132 steps)")
    ax.plot(xh2, dh2, "s-", color=C_REGION, label="half2 (179 new steps, other tasks)")
    ax.axhline(0, color=C_DENSE, lw=0.8)
    ax.set_xscale("log", base=2); ax.set_xticks([0.35, 1, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["ctr\nonly", "1", "4", "8", "16", "32", "64"], fontsize=8.5)
    ax.annotate("median(L)=48", xy=(47.6, -6), xytext=(9, -9.6), fontsize=8, color=C_TREE, arrowprops=dict(arrowstyle="-", color=C_TREE, lw=0.7))
    ax.annotate("mean(L)=79", xy=(79.3, -2), xytext=(60, -5.2), fontsize=8, color=C_TREE, arrowprops=dict(arrowstyle="-", color=C_TREE, lw=0.7))
    ax.set_xlabel("anchor A  (α = 0.25, budget 1024;  λ = A^0.25)"); ax.set_ylabel("agree − agree(A = 16), steps"); ax.set_ylim(-11, 7)
    ax.set_title("(b) A: no consistent effect from 1 to 64"); ax.legend(loc="lower left", fontsize=8.5)
    save(fig, "fig22_online_scans")


# fig23: robustness of the offline metric to seeds, and of the online metric to request order
def fig23():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    names = ["ctr", "A=1", "A=4", "A=8", "A=16", "A=32", "med", "A=64", "mean", "α=.125", "α=.375", "α=.5"]
    seed_m = [0.275, 0.267, 0.262, 0.259, 0.253, 0.245, 0.243, 0.237, 0.236, 0.232, 0.216, 0.188]
    seed_s = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0005, 0.002, 0.002]
    det = [0.277, 0.269, 0.266, 0.263, 0.259, 0.252, 0.249, 0.245, 0.243, 0.242, 0.225, 0.200]
    x = np.arange(len(names))
    ax.errorbar(x, seed_m, yerr=np.array(seed_s) * 3, fmt="o", color=C_REGION, capsize=3, label="random decode-step subsample, 5 seeds (mean ± 3 std)")
    ax.plot(x, det, "_", ms=14, mew=2, color=C_DENSE, label="fixed subsample (every 32nd step + last 4)")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5, rotation=30, ha="right"); ax.set_ylabel("offline mass, B = 1024"); ax.set_ylim(0.17, 0.29)
    ax.set_title("(a) offline metric: ordering identical under every seed"); ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    streams = ["original\norder", "repeat,\nsame order", "mixed into\n190-step run", "shuffled\norder"]
    a16 = [54, 54, 51, 56]; a8 = [54, None, 55, 55]
    ax.plot(range(4), a16, "o-", color=C_REGION, label="A = 16")
    ax.plot([0, 2, 3], [54, 55, 55], "s--", color=C_TREE, label="A = 8")
    ax.axhspan(51, 57, color=C_GRAY, alpha=0.15, lw=0)
    ax.set_xticks(range(4)); ax.set_xticklabels(streams, fontsize=8.5); ax.set_ylim(44, 62); ax.set_ylabel("agree (of 132 steps)")
    ax.set_title("(b) online: same config, different request order"); ax.legend(loc="upper left", fontsize=8.5)
    ax.text(1.5, 45.5, "same order → identical outputs (0 flips);\nother orders → ~7% of steps flip (batching)", ha="center", fontsize=8, color="#555555")
    save(fig, "fig23_seed_stream_robustness")


# fig24: whether the model's own generated tokens stay attendable (6.25% KV budget)
def fig24():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    names = ["ours\n(generated tokens\nalways attended)", "ours\n(generated tokens\nremoved)", "Quest", "BlockSparse"]
    vals = [41.2, 19.8, 21.3, 23.4]; cols = [C_REGION, C_BUG, C_FIXED, C_FIXED]
    ax.bar(range(4), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.9, f"{v:.1f}%", ha="center")
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8.8); ax.set_ylim(0, 50); ax.set_ylabel("accuracy (%)")
    ax.set_title("6.25% KV budget: the model must be able to read what it has already written")
    ax.annotate("", xy=(1, 21.5), xytext=(1, 40.5), arrowprops=dict(arrowstyle="<->", color=C_BUG, lw=1.2))
    ax.text(1.12, 31, "−21.4 points\nfrom dropping the\ngenerated tokens", fontsize=8.5, color=C_BUG, va="center")
    ax.axhspan(19.8, 23.4, color=C_GRAY, alpha=0.18, lw=0)
    ax.text(2.5, 17.3, "baselines: 21–23%", ha="center", fontsize=8.5, color="#555555")
    save(fig, "fig24_output_tokens")


if __name__ == "__main__":
    for f in (fig18, fig19, fig20, fig21, fig22, fig23, fig24):
        f()
