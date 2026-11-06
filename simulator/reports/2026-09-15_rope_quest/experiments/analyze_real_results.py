"""Task-balanced paired analysis of frozen-real-K RoPE interventions.

No pandas or scipy dependency. Aggregate layer/seed cells within capture, average
duplicate prompt captures, average prompts within task, then weight tasks equally.
Bootstrap tasks jointly across every condition so all reported contrasts are paired.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PRIMARY_METRICS = ["range_sum", "pooled_width", "pooled_gap", "perhead_width", "perhead_gap"]
BANDS = [f"band_{j}_{j + 7}" for j in range(0, 64, 8)]
CURVE_METRICS = [
    "range_sum", "pooled_center", "pooled_width", "pooled_bound", "pooled_true_max",
    "pooled_gap", "perhead_width", "perhead_bound", "perhead_true_max", "perhead_gap",
    *BANDS,
]
COMMON_ABS = "pooled_bound_mean_abs_change_vs_original"
FIT_LENGTHS = [8, 16, 32, 64, 128, 256]


def sha256(path):
    """Fingerprint an input or this analysis source."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    """Convert numpy scalars and unavailable logarithmic statistics for strict JSON."""
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def interval(samples):
    """Percentile 95% interval; require every resample to have a defined estimand."""
    if not np.isfinite(samples).all():
        return np.nan, np.nan
    return np.quantile(samples, [0.025, 0.975])


def stat_columns(name, estimate, samples):
    low, high = interval(samples)
    return {name: float(estimate), f"{name}_ci_low": float(low), f"{name}_ci_high": float(high)}


def log_slope(curves, lengths):
    """OLS slope of log arithmetic mean metric on log length; final axis is length."""
    x = np.log(np.asarray(lengths, dtype=float))
    x -= x.mean()
    valid = np.all(curves > 0, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.sum(np.log(curves) * x, axis=-1) / np.sum(x * x)
    return np.where(valid, slope, np.nan)


def ratio(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator != 0, numerator / denominator, np.nan)


def read_and_aggregate(path, manifest):
    """Validate a complete balanced grid and return task×mode×length×metric arrays."""
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No input rows")
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    lengths = sorted({int(r["length"]) for r in rows})
    if "original" not in modes or not set(FIT_LENGTHS) <= set(lengths):
        raise ValueError("Missing baseline or primary fitting lengths")
    metrics = CURVE_METRICS + ([COMMON_ABS] if COMMON_ABS in rows[0] else [])
    mid, lid = {m: i for i, m in enumerate(modes)}, {l: i for i, l in enumerate(lengths)}
    metadata, groups, cells, seen = {}, defaultdict(list), defaultdict(set), set()
    for row in rows:
        cap = row["capture"]
        meta = {k: row[k] for k in ("prompt_hash", "task", "site")}
        if cap in metadata and metadata[cap] != meta:
            raise ValueError(f"Inconsistent capture metadata: {cap}")
        metadata[cap] = meta
        layer, seed, length = (int(row[k]) for k in ("layer", "seed", "length"))
        key = (cap, layer, seed, length, row["mode"])
        if key in seen:
            raise ValueError(f"Duplicate raw cell: {key}")
        seen.add(key)
        vals = np.asarray([float(row[m]) for m in metrics])
        if not np.isfinite(vals).all():
            raise ValueError(f"Nonfinite raw metrics: {key}")
        groups[(cap, row["mode"], length)].append(vals)
        cells[(cap, row["mode"], length)].add((layer, seed))
    all_layer_seed = set.union(*cells.values())
    expected_layers = set(map(int, manifest.get("args", {}).get("layers", "").split(","))) if manifest else set()
    expected_seeds = set(map(int, manifest.get("args", {}).get("seeds", "").split(","))) if manifest else set()
    if manifest and all_layer_seed != {(l, s) for l in expected_layers for s in expected_seeds}:
        raise ValueError("Raw layer/seed grid disagrees with manifest")
    expected_captures = {c["capture"] for c in manifest.get("captures", [])} if manifest else set()
    if expected_captures and set(metadata) != expected_captures:
        raise ValueError("Raw capture set disagrees with manifest")
    capture_arrays = {}
    for cap in metadata:
        array = np.empty((len(modes), len(lengths), len(metrics)))
        for mode in modes:
            for length in lengths:
                group = (cap, mode, length)
                if cells[group] != all_layer_seed:
                    raise ValueError(f"Incomplete or unequal paired layer/seed grid: {group}")
                array[mid[mode], lid[length]] = np.mean(groups[group], axis=0)
        capture_arrays[cap] = array
    prompt_caps, prompt_task = defaultdict(list), {}
    for cap, meta in metadata.items():
        prompt = meta["prompt_hash"]
        if prompt in prompt_task and prompt_task[prompt] != meta["task"]:
            raise ValueError("Identical prompt assigned to different tasks; explicit policy needed")
        prompt_task[prompt] = meta["task"]
        prompt_caps[prompt].append(cap)
    task_prompts = defaultdict(list)
    for prompt, caps in prompt_caps.items():
        task_prompts[prompt_task[prompt]].append(np.mean([capture_arrays[c] for c in caps], axis=0))
    tasks = sorted(task_prompts)
    task_array = np.stack([np.mean(task_prompts[t], axis=0) for t in tasks])
    inventory = {
        "raw_rows": len(rows), "captures": len(metadata), "unique_prompts": len(prompt_caps),
        "tasks": len(tasks), "task_names": tasks,
        "layers": sorted({l for l, _ in all_layer_seed}), "seeds": sorted({s for _, s in all_layer_seed}),
        "spans_per_cell": sorted({int(r["spans"]) for r in rows}),
        "queries_per_cell": sorted({int(r["query_count"]) for r in rows}),
        "duplicate_prompts": [caps for caps in prompt_caps.values() if len(caps) > 1],
        "unique_prompts_per_task": {t: len(task_prompts[t]) for t in tasks},
    }
    return rows, task_array, tasks, modes, lengths, metrics, inventory


def write_csv(path, rows):
    """Write tidy results, using empty fields for undefined log slopes."""
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(finite(row) for row in rows)


def analyze(input_path, output_dir, manifest_path, n_bootstrap, seed):
    manifest = json.loads(manifest_path.read_text()) if manifest_path and manifest_path.exists() else {}
    if manifest and manifest.get("elapsed_seconds") is None:
        raise ValueError("Manifest does not mark the experiment complete")
    raw, data, tasks, modes, lengths, metrics, inventory = read_and_aggregate(input_path, manifest)
    nm, nl, nk = len(modes), len(lengths), len(metrics)
    orig = modes.index("original")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(tasks), size=(n_bootstrap, len(tasks)))
    weights = np.stack([np.bincount(d, minlength=len(tasks)) for d in draws]) / len(tasks)
    means = data.mean(axis=0)
    boot = (weights @ data.reshape(len(tasks), -1)).reshape(n_bootstrap, nm, nl, nk)
    fit_idx = [lengths.index(l) for l in FIT_LENGTHS]
    curve_rows, beta_rows, slope_rows, task_rows, common = [], [], [], [], []
    for mi, mode in enumerate(modes):
        for ki, metric in enumerate(metrics):
            if metric == COMMON_ABS and not mode.startswith("common_shift"):
                continue  # Stored zeros outside common-rotation controls are placeholders.
            for li, length in enumerate(lengths):
                est, samples = means[mi, li, ki], boot[:, mi, li, ki]
                base, bs = means[orig, li, ki], boot[:, orig, li, ki]
                row = {"mode": mode, "metric": metric, "length": length}
                row.update(stat_columns("estimate", est, samples))
                row.update(stat_columns("delta_vs_original", est - base, samples - bs))
                row.update(stat_columns("percent_change", 100 * ratio(est - base, base), 100 * ratio(samples - bs, bs)))
                row["tasks_with_negative_delta"] = int(np.sum(data[:, mi, li, ki] < data[:, orig, li, ki]))
                row["tasks_with_positive_delta"] = int(np.sum(data[:, mi, li, ki] > data[:, orig, li, ki]))
                curve_rows.append(row)
                for ti, task in enumerate(tasks):
                    task_rows.append({"task": task, "mode": mode, "length": length,
                                      "metric": metric, "estimate": data[ti, mi, li, ki]})
            if metric not in PRIMARY_METRICS + BANDS:
                continue
            curve, bc = means[mi, fit_idx, ki], boot[:, mi, :, ki][:, fit_idx]
            curve0, bc0 = means[orig, fit_idx, ki], boot[:, orig, :, ki][:, fit_idx]
            beta, bb = log_slope(curve, FIT_LENGTHS), log_slope(bc, FIT_LENGTHS)
            b0, bb0 = log_slope(curve0, FIT_LENGTHS), log_slope(bc0, FIT_LENGTHS)
            row = {"mode": mode, "metric": metric, "length_min": 8, "length_max": 256}
            row.update(stat_columns("beta", beta, bb))
            row.update(stat_columns("delta_beta_vs_original", beta - b0, bb - bb0))
            ep, eb = log_slope(curve[[0, -1]], [8, 256]), log_slope(bc[:, [0, -1]], [8, 256])
            ep0, eb0 = log_slope(curve0[[0, -1]], [8, 256]), log_slope(bc0[:, [0, -1]], [8, 256])
            row.update(stat_columns("endpoint_exponent", ep, eb))
            row.update(stat_columns("delta_endpoint_exponent", ep - ep0, eb - eb0))
            row.update(stat_columns("growth_ratio_256_over_8", ratio(curve[-1], curve[0]), ratio(bc[:, -1], bc[:, 0])))
            per_task = log_slope(data[:, mi, :, ki][:, fit_idx], FIT_LENGTHS)
            per_task0 = log_slope(data[:, orig, :, ki][:, fit_idx], FIT_LENGTHS)
            row.update(stat_columns("mean_task_beta", per_task.mean(), weights @ per_task))
            row.update(stat_columns("delta_mean_task_beta", (per_task - per_task0).mean(), weights @ (per_task - per_task0)))
            for j, length in ((0, 8), (-1, 256)):
                row.update(stat_columns(f"value_L{length}", curve[j], bc[:, j]))
                row.update(stat_columns(f"delta_L{length}", curve[j] - curve0[j], bc[:, j] - bc0[:, j]))
                row.update(stat_columns(f"percent_change_L{length}", 100 * ratio(curve[j] - curve0[j], curve0[j]),
                                        100 * ratio(bc[:, j] - bc0[:, j], bc0[:, j])))
            beta_rows.append(row)
            for li in range(nl - 1):
                pair_idx = [li, li + 1]
                pair_lengths = [lengths[i] for i in pair_idx]
                b = log_slope(means[mi, pair_idx, ki], pair_lengths)
                bboot = log_slope(boot[:, mi, :, ki][:, pair_idx], pair_lengths)
                bbase = log_slope(means[orig, pair_idx, ki], pair_lengths)
                bbaseboot = log_slope(boot[:, orig, :, ki][:, pair_idx], pair_lengths)
                row = {"mode": mode, "metric": metric, "length_from": lengths[li], "length_to": lengths[li + 1]}
                row.update(stat_columns("local_exponent", b, bboot))
                row.update(stat_columns("delta_vs_original", b - bbase, bboot - bbaseboot))
                slope_rows.append(row)
    for mode in (m for m in modes if m.startswith("common_shift")):
        rs = [r for r in raw if r["mode"] == mode]
        item = {"mode": mode, "max_logit_invariance_abs_error": max(float(r["logit_invariance_max_abs_error"]) for r in rs),
                "max_mean_pooled_true_max_change": 0.0, "max_mean_perhead_true_max_change": 0.0,
                "curves": [r for r in curve_rows if r["mode"] == mode and r["metric"] in ["pooled_bound", "pooled_true_max", "pooled_gap", "perhead_bound", "perhead_true_max", "perhead_gap", COMMON_ABS]]}
        original_rows = {(r["capture"], r["layer"], r["seed"], r["length"]): r for r in raw if r["mode"] == "original"}
        for r in rs:
            r0 = original_rows[(r["capture"], r["layer"], r["seed"], r["length"])]
            for pooling in ["pooled", "perhead"]:
                k = f"max_mean_{pooling}_true_max_change"
                item[k] = max(item[k], abs(float(r[f"{pooling}_true_max"]) - float(r0[f"{pooling}_true_max"])))
        common.append(item)
    qc = {"minimum_raw_mean_pooled_gap": min(float(r["pooled_gap"]) for r in raw),
          "minimum_raw_mean_perhead_gap": min(float(r["perhead_gap"]) for r in raw),
          "max_logit_invariance_abs_error": max((r["max_logit_invariance_abs_error"] for r in common), default=None),
          "complete_balanced_pairing": True}
    definitions = {
        "range_sum": "Sum of coordinate ranges over 128 dimensions; mean across spans and 4 KV heads; no query weights.",
        "band_j_k": "Mean of half-sum of ranges for split-half pairs (j,j+64), averaged within named band, spans, and KV heads; no query weights.",
        "pooled_width": "For each KV head, average its 8 query heads first; sum abs(qbar_d)*(max K_d-min K_d)/2, divide sqrt128, then mean over query calls, spans and KV heads.",
        "pooled_bound": "qbar dot box center /sqrt128 plus pooled width, with the same means.",
        "pooled_true_max": "Maximum token logit per pooled query and KV head, followed by averaging; not maximum after head averaging.",
        "pooled_gap": "Pooled bound minus that condition's pooled true maximum, recomputed after every intervention.",
        "perhead_width_bound_true_max_gap": "Same definitions separately for all 32 unpooled query heads, then averaged across heads, calls, spans.",
        COMMON_ABS: "For common rotations only: mean absolute change of each span/query/KV-head pooled bound; avoids cancellation of signed mean changes. Placeholder zeros elsewhere excluded.",
        "primary_beta": "OLS slope of log(task-balanced arithmetic mean metric) on log L for L=8,16,32,64,128,256. Bootstrap recomputes the entire mean curve and slope.",
        "mean_task_beta": "Sensitivity estimand: fit one slope to each task's arithmetic mean curve, then average slopes equally over tasks.",
        "endpoint_exponent": "log(mean metric at256 / mean metric at8) / log32; distinct from the six-point OLS slope.",
        "local_exponent": "log(mean metric at L2 / mean metric at L1) / log(L2/L1). Undefined if either mean is nonpositive; emitted as null/empty.",
        "percent_change": "100*(condition task-balanced mean minus original task-balanced mean)/original task-balanced mean, recomputed in paired bootstrap samples.",
        "ci": "Percentile95% intervals from joint task resampling with replacement. Conditional on these selected tasks and archived trajectories; not a causal population trial.",
    }
    summary = {
        "input_sha256": sha256(input_path), "analysis_script_sha256": sha256(__file__),
        "manifest_sha256": sha256(manifest_path) if manifest_path and manifest_path.exists() else None,
        "input": str(input_path.resolve()), "experiment_script_sha256": manifest.get("script_sha256"),
        "inventory": inventory, "modes": modes, "lengths": lengths,
        "aggregation": "Equal layer/seed mean within capture; equal captures within exact prompt_hash; equal unique prompts within task; equal tasks.",
        "bootstrap": {"resamples": n_bootstrap, "seed": seed, "unit": "task", "paired": True, "interval": "percentile95%"},
        "definitions": definitions, "quality_checks": qc,
        "primary_results": [r for r in beta_rows if r["metric"] in PRIMARY_METRICS],
        "band_exponents": [r for r in beta_rows if r["metric"] in BANDS],
        "common_rotation_controls": common,
        "interpretation_limits": [
            "Phase interventions act on frozen, post-RoPE real keys while holding the recorded query fixed; they are not a model forward without RoPE.",
            "Earlier-layer positional effects remain in captured key content. Lower width need not imply lower gap because transformed true maxima change too.",
            "Common rotations of both Q and K preserve logits; bound changes under this control identify axis-aligned box sensitivity to coordinate frame.",
            "Captures were selected sparse-failure/dense-success cases and use sparse-run query trajectories. They do not estimate end-to-end task accuracy.",
            "Only three query calls and sampled windows per seed were summarized in each raw row. Task bootstrap does not recover within-window/query sampling uncertainty or uncertainty over unseen model checkpoints.",
            "Short-length windows use starts valid for the 1024-token window to maintain nested matching. This is distinct from original independent per-length sampling.",
            "Intervals across many conditions, lengths, and metrics have no multiplicity correction; prioritize the predefined pooled/perhead width and gap comparison over 8–256.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary_curves.csv", curve_rows)
    write_csv(output_dir / "summary_betas.csv", beta_rows)
    write_csv(output_dir / "summary_local_slopes.csv", slope_rows)
    write_csv(output_dir / "summary_task_curves.csv", task_rows)
    (output_dir / "summary.json").write_text(json.dumps(finite(summary), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"inventory": inventory, "quality_checks": qc, "output": str(output_dir)}, indent=2))
    for r in summary["primary_results"]:
        if r["metric"] in ["pooled_width", "pooled_gap", "perhead_gap"] and r["mode"] in ["original", "collapse_all", "phase_speed_025", "phase_speed_4"]:
            print(f"{r['mode']:22s} {r['metric']:15s} beta={r['beta']:.5f} delta_beta={r['delta_beta_vs_original']:+.5f} L256change={r['percent_change_L256']:+.3f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()
    if args.bootstrap < 1:
        ap.error("--bootstrap must be positive")
    analyze(args.input, args.output, args.manifest, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
