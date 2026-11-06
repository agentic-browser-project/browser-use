"""Paired task bootstrap for fixed-budget selection on archived real Q/K.

Average calls within layer, layers within capture, duplicate captures within
prompt, unique prompts within task, and tasks equally. All contrasts reuse the
same task draws. Native and fixed16 collapse targets are explicitly different.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = ["retained_mass", "selected_token_jaccard_vs_original", "used_tokens",
           "selected_mean_chunk_length", "selected_long_token_fraction"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(prefix, estimate, boot):
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {prefix: float(estimate), f"{prefix}_ci_low": float(lo), f"{prefix}_ci_high": float(hi)}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(input_path, output_dir, manifest_path, n_bootstrap, seed):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("elapsed_seconds") is None:
        raise ValueError("Selection experiment incomplete")
    with input_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    conditions = sorted({(r["partition"], r["mode"], r["method"]) for r in rows})
    condition_idx = {c: i for i, c in enumerate(conditions)}
    groups, layer_calls, meta, seen = defaultdict(list), defaultdict(set), {}, set()
    for r in rows:
        cap, layer, call = r["capture"], int(r["layer"]), int(r["call"])
        condition = (r["partition"], r["mode"], r["method"])
        cell = (cap, condition, layer, call)
        if cell in seen:
            raise ValueError(f"Duplicate row: {cell}")
        seen.add(cell)
        identity = (r["prompt_hash"], r["task"])
        if cap in meta and meta[cap] != identity:
            raise ValueError("Inconsistent capture identity")
        meta[cap] = identity
        values = np.asarray([float(r[m]) for m in METRICS])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite primary selection metrics")
        groups[(cap, condition, layer)].append(values)
        layer_calls[(cap, condition, layer)].add(call)
    if set(meta) != {c["capture"] for c in manifest["captures"]}:
        raise ValueError("Manifest capture set mismatch")
    layers = sorted({int(r["layer"]) for r in rows})
    caps = {}
    for cap in meta:
        expected_calls = layer_calls[(cap, conditions[0], layers[0])]
        values = np.empty((len(conditions), len(METRICS)))
        for ci, condition in enumerate(conditions):
            if any(layer_calls[(cap, condition, layer)] != expected_calls for layer in layers):
                raise ValueError(f"Incomplete paired calls/layers: {cap} {condition}")
            values[ci] = np.mean([np.mean(groups[(cap, condition, layer)], axis=0) for layer in layers], axis=0)
        caps[cap] = values
    prompt_caps, prompt_task = defaultdict(list), {}
    for cap, (prompt, task) in meta.items():
        if prompt in prompt_task and prompt_task[prompt] != task:
            raise ValueError("Prompt assigned to two tasks")
        prompt_caps[prompt].append(cap)
        prompt_task[prompt] = task
    task_prompts = defaultdict(list)
    for prompt, capture_list in prompt_caps.items():
        task_prompts[prompt_task[prompt]].append(np.mean([caps[c] for c in capture_list], axis=0))
    tasks = sorted(task_prompts)
    array = np.stack([np.mean(task_prompts[t], axis=0) for t in tasks])
    means = array.mean(0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(tasks), size=(n_bootstrap, len(tasks)))
    weights = np.stack([np.bincount(d, minlength=len(tasks)) for d in draws]) / len(tasks)
    boot = (weights @ array.reshape(len(tasks), -1)).reshape(n_bootstrap, len(conditions), len(METRICS))
    result_rows, task_rows, partition_rows = [], [], []
    for ci, (partition, mode, method) in enumerate(conditions):
        baseline = condition_idx[(partition, "original", method)]
        quest = condition_idx[(partition, mode, "quest")]
        for ki, metric in enumerate(METRICS):
            row = {"partition": partition, "mode": mode, "method": method, "metric": metric}
            row.update(stats("estimate", means[ci, ki], boot[:, ci, ki]))
            delta = means[ci, ki] - means[baseline, ki]
            paired = boot[:, ci, ki] - boot[:, baseline, ki]
            row.update(stats("delta_vs_original", delta, paired))
            row.update(stats("delta_vs_quest_same_target", means[ci, ki] - means[quest, ki], boot[:, ci, ki] - boot[:, quest, ki]))
            row["tasks_with_positive_delta_vs_original"] = int((array[:, ci, ki] > array[:, baseline, ki]).sum())
            row["tasks_with_negative_delta_vs_original"] = int((array[:, ci, ki] < array[:, baseline, ki]).sum())
            row["same_logits_as_original"] = mode != "collapse_all"
            result_rows.append(row)
            for ti, task in enumerate(tasks):
                task_rows.append({"task": task, "partition": partition, "mode": mode, "method": method,
                                  "metric": metric, "estimate": float(array[ti, ci, ki])})
            if partition == "native" and mode != "collapse_all":
                other = condition_idx[("fixed16", mode, method)]
                r = {"mode": mode, "method": method, "metric": metric,
                     "contrast": "native_minus_fixed16"}
                r.update(stats("delta", means[ci, ki] - means[other, ki], boot[:, ci, ki] - boot[:, other, ki]))
                partition_rows.append(r)
    controls = []
    for partition in sorted({r["partition"] for r in rows}):
        for mode in sorted({r["mode"] for r in rows if r["mode"].startswith("common_shift")}):
            rset = [r for r in rows if r["partition"] == partition and r["mode"] == mode]
            oracle = [r for r in rset if r["method"] == "exact_density_greedy"]
            controls.append({"partition": partition, "mode": mode,
                "max_logit_abs_error": max(float(r["logit_invariance_max_abs_error"]) for r in rset),
                "max_chunk_mass_abs_error": max(float(r["mass_invariance_max_abs_error"]) for r in rset),
                "exact_density_rows": len(oracle),
                "exact_density_rows_with_changed_selection": sum(float(r["selected_token_jaccard_vs_original"]) < 1 - 1e-12 for r in oracle),
                "exact_density_min_jaccard": min(float(r["selected_token_jaccard_vs_original"]) for r in oracle)})
    summary = {
        "input_sha256": digest(input_path), "manifest_sha256": digest(manifest_path),
        "analysis_script_sha256": digest(__file__), "experiment_script_sha256": manifest["script_sha256"],
        "inventory": {"rows": len(rows), "captures": len(meta), "unique_prompts": len(prompt_caps),
                      "tasks": len(tasks), "task_names": tasks, "layers": layers,
                      "budgets": sorted({int(r["budget"]) for r in rows})},
        "aggregation": "Equal calls within layer, layers within capture, captures within exact prompt, unique prompts within task, and tasks equally.",
        "bootstrap": {"resamples": n_bootstrap, "seed": seed, "unit": "task", "paired": True, "interval": "percentile95%"},
        "definitions": {
            "retained_mass": "Sum of exact prefill-only token attention mass retained by selection. Softmax computed separately for each full query head, then averaged across32 heads. Delta0.01 means1 percentage point.",
            "selected_token_jaccard_vs_original": "Token-weighted intersection/union of current and original selected chunks for the same partition/query/layer/method; not unweighted chunk Jaccard.",
            "quest": "Mean over KV heads of box support function using within-GQA-group mean query.",
            "width_alpha025_anchor16": "Pooled-query center plus (16/L)^0.25 times pooled-query width, averaged across KV heads.",
            "exact_density_greedy": "Greedy score uses actual chunk mass per token. This is a reference heuristic, not an optimal0–1 knapsack solver.",
            "admission": "Sort chunks descending by score, add whole chunks only when they fit remaining2048-token budget, continue scanning. No pinned/system-floor/always-include tokens.",
            "native": "The archived capture's original chunk spans, not the later region-aware6/64 and64/256 reconstruction.",
            "common_rotation": "Same coordinate rotation for all keys and queries; exact target logits/mass invariant up to floating-point error.",
            "collapse_all": "Remove phase offsets relative to each chunk's first token. Target changes, and the transformation itself depends on partition. Do not compare native/fixed16 collapse as identical-target selection.",
        },
        "results": result_rows, "same_target_partition_contrasts": partition_rows,
        "invariance_and_tie_checks": controls,
        "limitations": [
            "Failure-selected archived trajectories; frozen sparse-run queries and prefill-only K. No inference about end-to-end accuracy or latency.",
            "Common-rotation Quest differences isolate representation-coordinate sensitivity while holding logits fixed; density-reference changes would indicate finite-precision/tie sensitivity.",
            "Collapse changes attention targets. Its retained mass is not directly comparable to original mass as if the task remained fixed.",
            "Task-bootstrap intervals are conditional on13 observed tasks and the selected calls; no multiplicity correction is applied.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "selection_summary.csv", result_rows)
    write_csv(output_dir / "selection_summary_tasks.csv", task_rows)
    write_csv(output_dir / "selection_summary_partitions.csv", partition_rows)
    (output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"inventory": summary["inventory"], "controls": controls}, indent=2))
    for r in result_rows:
        if r["metric"] in ["retained_mass", "selected_token_jaccard_vs_original"]:
            print(f"{r['partition']:7s} {r['mode']:19s} {r['method']:25s} {r['metric']:36s} {r['estimate']:.6f} delta={r['delta_vs_original']:+.6f} [{r['delta_vs_original_ci_low']:+.6f},{r['delta_vs_original_ci_high']:+.6f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    analyze(args.input, args.output, args.manifest, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
