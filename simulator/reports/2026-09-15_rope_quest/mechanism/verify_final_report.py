"""Independent raw-CSV aggregation and headline checks; does not import analysis code."""
from pathlib import Path
from collections import defaultdict
import csv, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

def read(name):
    return list(csv.DictReader((RESULTS / name).open()))

def aggregate(rows, condition_columns, metrics):
    cells, meta = defaultdict(list), {}
    for r in rows:
        condition = tuple(r[k] for k in condition_columns)
        cells[(r["capture"], condition)].append([float(r[m]) for m in metrics])
        meta[r["capture"]] = (r["prompt_hash"], r["task"])
    prompts = defaultdict(list)
    for (capture, condition), values in cells.items():
        prompt, task = meta[capture]
        prompts[(task, prompt, condition)].append(np.mean(values, axis=0))
    tasks = defaultdict(list)
    for (task, prompt, condition), values in prompts.items():
        tasks[(task, condition)].append(np.mean(values, axis=0))
    by_task = {k: np.mean(v, axis=0) for k, v in tasks.items()}
    task_names = sorted({k[0] for k in by_task})
    conditions = sorted({k[1] for k in by_task})
    means = {c: np.mean([by_task[t, c] for t in task_names], axis=0) for c in conditions}
    return means, by_task, task_names

raw = read("paired_metrics.csv")
curves = read("summary_curves.csv")
metrics = sorted({r["metric"] for r in curves})
means, tasks, task_names = aggregate(raw, ["mode", "length"], metrics)
errors = [abs(float(r["estimate"]) - means[r["mode"], r["length"]][metrics.index(r["metric"])]) for r in curves]
assert max(errors) < 1e-10

selection = read("selection_metrics.csv")
summary = read("selection_summary.csv")
smetrics = sorted({r["metric"] for r in summary})
smeans, stasks, stask_names = aggregate(selection, ["partition", "mode", "method"], smetrics)
serrors = [abs(float(r["estimate"]) - smeans[r["partition"], r["mode"], r["method"]][smetrics.index(r["metric"])]) for r in summary]
assert max(serrors) < 1e-10

# Independently resample task arrays. Same declared seed permits an exact audit.
draws = np.random.default_rng(20260915).integers(0, len(task_names), (5000, len(task_names)))
ci_checks = {}
for metric in ["pooled_width", "pooled_gap"]:
    j = metrics.index(metric)
    original = np.array([tasks[t, ("original", "256")][j] for t in task_names])
    altered = np.array([tasks[t, ("collapse_all", "256")][j] for t in task_names])
    draws_original = original[draws].mean(1)
    draws_altered = altered[draws].mean(1)
    ci = np.quantile(100 * (draws_altered / draws_original - 1), [.025, .975])
    saved = next(r for r in curves if r["mode"] == "collapse_all" and r["length"] == "256" and r["metric"] == metric)
    assert np.max(np.abs(ci - [float(saved["percent_change_ci_low"]),float(saved["percent_change_ci_high"])])) < 1e-10
    ci_checks[metric+"_percent_change_ci"] = ci.tolist()

fit_lengths = [8,16,32,64,128,256]
j = metrics.index("pooled_width")
beta_boot = {}
for mode in ["original", "collapse_all"]:
    values = np.array([[tasks[t, (mode, str(n))][j] for n in fit_lengths] for t in task_names])
    boot_curves = values[draws].mean(1)
    beta_boot[mode] = np.polyfit(np.log(fit_lengths), np.log(boot_curves).T, 1)[0]
beta_delta_ci = np.quantile(beta_boot["collapse_all"] - beta_boot["original"], [.025,.975])
b = next(r for r in read("summary_betas.csv") if r["mode"]=="collapse_all" and r["metric"]=="pooled_width")
assert np.max(np.abs(beta_delta_ci - [float(b["delta_beta_vs_original_ci_low"]),float(b["delta_beta_vs_original_ci_high"])])) < 1e-10
ci_checks["width_slope_paired_delta_ci"] = beta_delta_ci.tolist()

assert stask_names == task_names
j = smetrics.index("retained_mass")
conditions = [("native","original","quest"),
              ("native","original","width_alpha025_anchor16"),
              ("native","common_shift_4093","quest")]
sampled = {c: np.array([stasks[t,c][j] for t in task_names])[draws].mean(1) for c in conditions}
for condition, label in [(conditions[1],"calibration_gain_pp"),(conditions[2],"common_shift4093_delta_pp")]:
    ci = 100*np.quantile(sampled[condition]-sampled[conditions[0]],[.025,.975])
    saved=next(r for r in summary if tuple(r[k] for k in ["partition","mode","method"])==condition and r["metric"]=="retained_mass")
    prefix = "delta_vs_quest_same_target" if condition==conditions[1] else "delta_vs_original"
    assert np.max(np.abs(ci - 100*np.array([float(saved[prefix+"_ci_low"]),float(saved[prefix+"_ci_high"])]))) < 1e-10
    ci_checks[label+"_ci"] = ci.tolist()

common = [r for r in selection if r["mode"].startswith("common_") and r["method"]=="exact_density_greedy"]
assert len(common)==1656
assert all(float(r["selected_token_jaccard_vs_original"])==1 for r in common)
output=dict(real_rows=len(raw),selection_rows=len(selection),task_count=len(task_names),
            curve_estimates_checked=len(curves),max_curve_error=max(errors),
            selection_estimates_checked=len(summary),max_selection_error=max(serrors),
            exact_density_common_rotation_cells=len(common),
            all_exact_density_common_selections_identical=True,
            max_window_logit_discrepancy=max(float(r["logit_invariance_max_abs_error"]) for r in raw),
            max_selection_chunk_mass_discrepancy=max(float(r["mass_invariance_max_abs_error"]) for r in selection),
            independent_bootstrap_checks=ci_checks)
(Path(__file__).parent/"final_verification.json").write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps(output,indent=2))
