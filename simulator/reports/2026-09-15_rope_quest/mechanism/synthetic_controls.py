"""Fixed-query synthetic controls for Quest/RoPE; numpy + matplotlib only.

Run: /tmp/context-composition-venv/bin/python synthetic_controls.py
All outputs are beside this script. No model/data downloads; no fitted parameters.
"""
from pathlib import Path
import csv
import json
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SEED, REPEATS, DIM, MAX_LENGTH = 20260915, 512, 128, 1024
LENGTHS = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
rng = np.random.default_rng(SEED)
q = rng.normal(size=DIM)
q /= np.linalg.norm(q)
theta = 5e6 ** (-np.arange(DIM // 2) * 2 / DIM)


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def rotate_pairs(x, angles):
    a, b = np.split(x, 2, axis=-1)
    ca, sa = np.cos(angles), np.sin(angles)
    return np.concatenate([a * ca - b * sa, a * sa + b * ca], axis=-1)


def score_prefixes(keys, query, lengths=LENGTHS):
    """keys [repeat, token, dim], query fixed for every prefix/repetition."""
    lo = np.minimum.accumulate(keys, axis=1)[:, lengths - 1, :]
    hi = np.maximum.accumulate(keys, axis=1)[:, lengths - 1, :]
    c = ((lo + hi) / 2) @ query
    w = ((hi - lo) / 2) @ np.abs(query)
    scores = keys @ query
    maxima = np.maximum.accumulate(scores, axis=1)[:, lengths - 1]
    means = np.cumsum(scores, axis=1)[:, lengths - 1] / lengths
    # Stable exact log mean exp; prefix-by-prefix to keep memory bounded.
    lme = np.stack([
        np.log(np.exp(scores[:, :n] - maxima[:, j:j+1]).mean(axis=1)) + maxima[:, j]
        for j, n in enumerate(lengths)
    ], axis=1)
    return dict(c=c, w=w, bound=c+w, true_max=maxima,
                gap=c+w-maxima, mean_logit=means, log_mean_exp=lme)


def summarize(name, arrays):
    rows = []
    for j, n in enumerate(LENGTHS):
        row = dict(condition=name, L=int(n), repeats=len(arrays["w"]))
        for k, values in arrays.items():
            row[k] = float(values[:, j].mean())
            row[k+"_se"] = float(values[:, j].std(ddof=1) / np.sqrt(len(values)))
        rows.append(row)
    return rows


def exponent(rows, key="w", start=8, end=256):
    sub = [r for r in rows if start <= r["L"] <= end and r[key] > 0]
    return float(np.polyfit(np.log([r["L"] for r in sub]),
                            np.log([r[key] for r in sub]), 1)[0])


all_rows = []
paired_rows = []
data = {}
angles = np.arange(MAX_LENGTH)[:, None] * theta
for condition in ["iid_gaussian_no_rope", "iid_gaussian_rope", "iid_uniform_no_rope"]:
    data[condition] = {k: [] for k in ["c", "w", "bound", "true_max", "gap", "mean_logit", "log_mean_exp"]}

# Pair no-RoPE and RoPE within each replicate. An isotropic Gaussian has the
# same joint law after arbitrary per-token orthogonal rotations, exactly.
for offset in range(0, REPEATS, 16):
    z = rng.normal(size=(16, MAX_LENGTH, DIM))
    a = score_prefixes(z, q)
    b = score_prefixes(rotate_pairs(z, angles), q)
    u = score_prefixes(rng.uniform(-np.sqrt(3), np.sqrt(3), z.shape), q)
    for name, result in zip(data, [a, b, u]):
        for k, value in result.items():
            data[name][k].append(value)
for name, arrays in data.items():
    data[name] = {k: np.concatenate(v) for k, v in arrays.items()}
    all_rows.extend(summarize(name, data[name]))
for j, n in enumerate(LENGTHS):
    row = dict(L=int(n))
    for key in ["w", "bound", "true_max", "gap"]:
        delta = data["iid_gaussian_rope"][key][:, j] - data["iid_gaussian_no_rope"][key][:, j]
        row[key+"_paired_delta"] = float(delta.mean())
        row[key+"_paired_delta_se"] = float(delta.std(ddof=1)/np.sqrt(REPEATS))
    paired_rows.append(row)

# Correlated 2D content has identically zero true logits but growing box slack.
x = rng.normal(size=(REPEATS, MAX_LENGTH))
anti = np.stack([x, -x], axis=-1)
anti_q = np.array([1., 1.]) / np.sqrt(2)
anti_scores = score_prefixes(anti, anti_q)
all_rows.extend(summarize("anticorrelated_gaussian_no_rope", anti_scores))

# Identical pre-RoPE content with a fixed query: only per-token rotation varies.
# Random phases vary the shared content across repetitions, not within a block.
phase = rng.uniform(0, 2*np.pi, size=(REPEATS, DIM//2))
common = np.concatenate([np.cos(phase), np.sin(phase)], axis=-1)
for name, rotate in [("identical_content_no_rope", False), ("identical_content_rope", True)]:
    blocks = []
    for off in range(0, REPEATS, 16):
        keys = np.broadcast_to(common[off:off+16, None], (16, MAX_LENGTH, DIM))
        if rotate:
            keys = rotate_pairs(keys, angles)
        blocks.append(score_prefixes(keys, q))
    arrays = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    all_rows.extend(summarize(name, arrays))
    data[name] = arrays

# Frequency bands for identical-content rotation; amplitudes = 1, same q.
band_rows = []
for band in range(8):
    ids = np.arange(8*band, 8*band+8)
    dims = np.r_[ids, ids + DIM//2]
    local_x = common[:, dims]
    local_q = q[dims]
    local_angles = np.arange(256)[:, None] * theta[ids]
    local_keys = rotate_pairs(local_x[:, None, :], local_angles)
    res = score_prefixes(local_keys, local_q, LENGTHS[LENGTHS <= 256])
    means = res["w"].mean(axis=0)
    for j, n in enumerate(LENGTHS[LENGTHS <= 256]):
        band_rows.append(dict(band=f"{ids[0]}-{ids[-1]}", L=int(n),
                              theta_min=float(theta[ids[-1]]),theta_max=float(theta[ids[0]]),
                              w=float(means[j])))

# Exact distribution formulas: uniform sample extrema and skewed exponential midrange.
exact_rows = []
for n in LENGTHS:
    harmonic = sum(1/k for k in range(1, int(n)+1))
    exact_rows.append(dict(L=int(n),
        uniform_expected_w=float(np.sqrt(3)*np.linalg.norm(q, ord=1)*(n-1)/(n+1)),
        exponential_expected_midrange=float((harmonic+1/n)/2),
        exponential_expected_sample_mean=1.0))

# A score-preserving common rotation must rotate both Q and every K.
base_keys = np.array([[1., -1.], [-1., 1.]])
base_q = np.array([1., 1.])
rotation_rows = []
for phi in np.linspace(0, np.pi/2, 361):
    keys_r = rotate_pairs(base_keys, np.array([phi]))
    q_r = rotate_pairs(base_q, np.array([phi]))
    res = score_prefixes(keys_r[None], q_r, np.array([2]))
    row = dict(degrees=float(phi*180/np.pi),
               max_logit_error=float(np.abs(keys_r@q_r-base_keys@base_q).max()))
    row.update({k: float(v[0, 0]) for k, v in res.items()})
    rotation_rows.append(row)

# Arbitrary slopes from pure rotation: local tangential, local radial, saturation.
local_rows = []
for name, th, query in [
    ("slow_tangential", 1e-5, np.array([0.,1.])),
    ("slow_radial", 1e-5, np.array([1.,0.])),
    ("fast_saturating", 1., np.array([1.,1.])/np.sqrt(2)),
]:
    keys = rotate_pairs(np.array([1.,0.]), np.arange(MAX_LENGTH)[:,None]*th)
    r = score_prefixes(keys[None], query)
    for j,n in enumerate(LENGTHS):
        local_rows.append(dict(condition=name,L=int(n),theta=th,
                               **{k:float(v[0,j]) for k,v in r.items()}))

write_csv("synthetic_summary.csv", all_rows)
write_csv("isotropic_gaussian_paired_delta.csv", paired_rows)
write_csv("identical_content_frequency_bands.csv", band_rows)
write_csv("exact_distribution_controls.csv", exact_rows)
write_csv("common_rotation_2d.csv", rotation_rows)
write_csv("single_pair_rotation_slopes.csv", local_rows)

conditions = sorted(set(r["condition"] for r in all_rows))
fits = {name: exponent([r for r in all_rows if r["condition"]==name])
        for name in conditions if name != "identical_content_no_rope"}
fits.update({name: exponent([r for r in local_rows if r["condition"]==name])
             for name in sorted(set(r["condition"] for r in local_rows))})
meta = dict(seed=SEED,repeats=REPEATS,dimension=DIM,lengths=LENGTHS.tolist(),
            q_norm=float(np.linalg.norm(q)),q_l1=float(np.linalg.norm(q,1)),
            python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
            fitted_loglog_w_slopes_8_to_256=fits,
            common_rotation_max_logit_error=max(r["max_logit_error"] for r in rotation_rows))
(OUT/"run_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")

plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
for name, label, color in [("iid_gaussian_no_rope","Gaussian, no RoPE","C0"),
                           ("iid_gaussian_rope","Gaussian, RoPE","C1"),
                           ("iid_uniform_no_rope","Uniform, no RoPE","C2")]:
    rows=[r for r in all_rows if r["condition"]==name]
    axes[0,0].plot(LENGTHS,[r["w"] for r in rows],"o-",label=label,color=color,ms=4)
axes[0,0].set(xscale="log",xlabel="Chunk length L",ylabel="Expected width w",
              title="A. Growth occurs without RoPE")
axes[0,0].legend(fontsize=8)
rows=[r for r in all_rows if r["condition"]=="anticorrelated_gaussian_no_rope"]
for key, label in [("bound","Quest bound = slack"),("true_max","True maximum")]:
    axes[0,1].plot(LENGTHS,[r[key] for r in rows],"o-",label=label,ms=4)
axes[0,1].set(xscale="log",xlabel="Chunk length L",ylabel="Score",
              title="B. Slack can grow while all true logits = 0")
axes[0,1].legend(fontsize=8)
for name,label in [("identical_content_no_rope","Identical content, no rotation"),
                   ("identical_content_rope","Identical content, RoPE")]:
    rows=[r for r in all_rows if r["condition"]==name]
    axes[1,0].plot(LENGTHS,[r["w"] for r in rows],"o-",label=label,ms=4)
axes[1,0].set(xscale="log",xlabel="Chunk length L",ylabel="Expected width w",
              title="C. Rotation alone is sufficient to create width")
axes[1,0].legend(fontsize=8)
axes[1,1].plot([r["degrees"] for r in rotation_rows],[r["bound"] for r in rotation_rows],label="Quest bound")
axes[1,1].plot([r["degrees"] for r in rotation_rows],[r["true_max"] for r in rotation_rows],label="Every true logit (unchanged)")
axes[1,1].set(xlabel="Common Q/K rotation (degrees)",ylabel="Score",
              title="D. The box depends on the coordinate axes")
axes[1,1].legend(fontsize=8)
fig.savefig(OUT/"synthetic_controls.png",dpi=180)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
for band in range(8):
    rows=[r for r in band_rows if r["band"]==f"{band*8}-{band*8+7}" and r["L"]>=8]
    ww=np.array([r["w"] for r in rows])
    axes[0].plot([r["L"] for r in rows],ww/ww[0],"o-",label=f"j = {band*8}-{band*8+7}",ms=3)
axes[0].set(xscale="log",yscale="log",xlabel="Chunk length L",ylabel="w(L) / w(8)",
            title="Pure identical-content rotation: relative band growth")
axes[0].legend(ncol=2,fontsize=8)
for name in sorted(set(r["condition"] for r in local_rows)):
    rows=[r for r in local_rows if r["condition"]==name and r["L"]>=8]
    ww=np.array([r["w"] for r in rows])
    axes[1].plot([r["L"] for r in rows],ww/ww[0],"o-",label=name.replace("_"," "),ms=3)
axes[1].set(xscale="log",yscale="log",xlabel="Chunk length L",ylabel="w(L) / w(8)",
            title="Pure rotation has no universal growth exponent")
axes[1].legend(fontsize=8)
fig.savefig(OUT/"frequency_and_local_slopes.png",dpi=180)
plt.close(fig)
print(json.dumps(meta,indent=2))
for name in conditions:
    rows=[r for r in all_rows if r["condition"]==name and r["L"] in [8,256]]
    print(name,[(r["L"],round(r["w"],6),round(r["true_max"],6),round(r["gap"],6)) for r in rows])
