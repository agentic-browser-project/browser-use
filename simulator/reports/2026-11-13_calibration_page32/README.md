# Offline action replay for the A × alpha sweep

This directory contains the code used by the active page32/max16 experiment on
spark00. The previous page64/max32 experiment is in
[`../2026-11-12_calibration_page64`](../2026-11-12_calibration_page64).
These are experiment snapshots with the original execution paths, not a portable
serving package. The inference code is unchanged from the running experiment.

## Entry points

| File | Responsibility |
| --- | --- |
| [`experiments/bootstrap.py`](experiments/bootstrap.py) | Load Qwen3-32B BF16, wait for prepared inputs, and invoke the replay driver. |
| [`experiments/run_replay.py`](experiments/run_replay.py) | Run the 24-setting grid, validate legacy continuations, reuse matching measurements, and save each completed step. |
| [`experiments/replay_attention.py`](experiments/replay_attention.py) | Compute calibrated chunk scores, select pages, and run sparse decoding with CUDA graphs and shared prompt KV. |
| [`experiments/fused_kv_gather.cu`](experiments/fused_kv_gather.cu) | Gather selected KV pages with a fused copy kernel. |
| [`experiments/layouts.py`](experiments/layouts.py) | Prepare per-configuration chunk summaries and physical page placement. |
| [`experiments/prepare_inputs.py`](experiments/prepare_inputs.py) | Freeze the 64-task inventory and preserve the historical numerical batch layout. |
| [`experiments/analyze_replay.py`](experiments/analyze_replay.py) | Aggregate complete tasks, produce full-pool rankings, and separately invoke exploratory subset searches. |
| [`experiments/subset_search.py`](experiments/subset_search.py) | Search 16-task and 32-task subsets under the declared target preference. This does not affect generation or full-pool rankings. |
| [`experiments/write_report.py`](experiments/write_report.py) | Generate the English report and ranking figures. |
| [`experiments/run_remote.sh`](experiments/run_remote.sh) | Host-side GPU launch command with a process lock. |
| [`experiments/finish_remote.sh`](experiments/finish_remote.sh) | Run rolling analysis from 16 completed tasks through 64. |
| [`experiments/collect_results.py`](experiments/collect_results.py) | Retrieve compact artifacts from spark00 without copying model or KV caches. |

## Configuration and metrics

[`settings.json`](settings.json) is the actual experiment configuration:

- Qwen/Qwen3-32B, BF16, thinking enabled, greedy decoding, at most 1024 output tokens.
- Page size 32, maximum chunk size 16, soft merge target 6, nominal page budget 12.5%.
- A in `{1, 2, 4, 8, 16, 32, 64, 128}` and alpha in `{0.125, 0.25, 0.5}`.
- 64 distinct WebVoyager tasks and 72 included steps. Three extra batch partners
  preserve numerical execution shapes and never enter rankings.

The primary metric requires both the full action-type sequence and extracted
target-index sequence to match dense. Dense must finish and produce a parseable
indexed action. An invalid or unfinished sparse output is a mismatch. The parser
does not normalize XML or alternate target-field names. This measures strict
output fidelity, not online task success.

The analyzer averages eligible steps within each task and then averages tasks.
It includes a task only after all its planned steps have a complete grid. A
step-weighted interim ranking instead averages all eligible completed steps,
including those from incomplete tasks. These are different aggregations.
`measured_pool` contains every eligible complete task without outcome-based
selection. `exploratory16` and `exploratory32` explicitly favor named parameter
settings and must not be reported as unbiased validation results.

See [`protocol.md`](protocol.md) for the frozen reuse and selection rules and
[`inputs/manifest.json`](inputs/manifest.json) for task IDs and input hashes.

## Code dependencies and execution paths

The runtime imports these unchanged modules, included in this branch:

- [`../2026-11-12_calibration32/experiments/run_replay.py`](../2026-11-12_calibration32/experiments/run_replay.py)
- [`../2026-11-11_semantic_ranking_grid/experiments/runtime.py`](../2026-11-11_semantic_ranking_grid/experiments/runtime.py)
- [`../2026-11-11_semantic_ranking_grid/experiments/replay_actions.py`](../2026-11-11_semantic_ranking_grid/experiments/replay_actions.py)
- [`../2026-11-11_semantic_ranking_grid/experiments/replay_grouped.py`](../2026-11-11_semantic_ranking_grid/experiments/replay_grouped.py)

The BrowSA parser, packing, query pooling, and selector come from
`agentic-browser-project/BrowserSparseAttention`, pinned at
`caf646eb2b014789ad5871092561c57da9245b61`. Their source hashes are in
[`../2026-11-11_semantic_ranking_grid/source_manifest.json`](../2026-11-11_semantic_ranking_grid/source_manifest.json).
That separate repository's source snapshot must be restored under
`/workspace/semantic_ranking_grid_20261111/source` with the paths in the manifest.
It is not vendored here.

| Repository directory | Existing container directory |
| --- | --- |
| `2026-11-13_calibration_page32` | `/workspace/calibration_page32_20261113` |
| `2026-11-12_calibration_page64` | `/workspace/calibration_page64_20261112` |
| `2026-11-12_calibration32` | `/workspace/calibration32_20261112` |
| `2026-11-11_semantic_ranking_grid` | `/workspace/semantic_ranking_grid_20261111` |

The host root `/home/shiqihe/workspace` is mounted as `/workspace` in the
`browsa-score-error-20261111` container. The model is at `/models/Qwen3-32B`.
Python is `/workspace/anchor_alpha_grid_20261110/.venv/bin/python`.

Full prompts, generated continuations, historical outputs, model weights, and
captured KV tensors remain experiment artifacts on spark00, outside this code
upload. In particular, page32 preparation requires the historical geometry,
replay JSON files, recent `inputs/tokenized.json`, and prompt captures referenced
by `prepare_inputs.py`. Git checkout alone does not provide these data.

In the existing prepared environment, the host launch commands are:

```bash
bash /home/shiqihe/workspace/calibration_page32_20261113/experiments/run_remote.sh
bash /home/shiqihe/workspace/calibration_page32_20261113/experiments/finish_remote.sh
```

They run as separate processes. The launcher resumes saved groups and the
postprocessor follows progress. Input preparation uses `prepare_inputs.py` inside
the container. After successful preparation, `ready.json` must record
`{"ready": true, "manifest_sha256": "<SHA256 of inputs/manifest.json>"}` before
bootstrap proceeds. Do not regenerate the frozen inventory with different input
artifacts when resuming this experiment.

## Runtime provenance and validation

[`runtime_manifest.json`](runtime_manifest.json) records hashes of the active
inference code and configuration. The upload was checked against the files on
spark00. The local collection script adds SSH options to avoid a stale connection
and is not part of model inference.

The measured environment uses NVIDIA GB10, Python 3.12.3, PyTorch
`2.10.0a0+b558c986e8.nv25.11`, Transformers 4.57.1, FlashAttention
`2.7.4.post1+25.11`, NumPy 2.1.0, pandas 2.3.3, SciPy 1.16.3, and Matplotlib 3.10.7.
These include NVIDIA container builds, not a generic PyPI requirements lock.

The driver checks fused KV copies bitwise, validates 147 legacy token prefixes,
and checks all legacy full continuations again before merging each reused step.
It stops on mismatches. Existing synthetic optimizer and aggregation checks are
recorded under [`validation/`](validation/). They are validation fixtures, not
measured benchmark outcomes.

## Target-ID comparison correction

The primary joint metric now treats integer IDs and equivalent integer strings
as the same target. For example, `8117` matches `"8117"`. This applies to every
configuration. The previous comparison is retained as `target_equal_raw`, while
`target_equal` uses `integer_target_ids_v2`. Action order, output completion, and
the action JSON parser are unchanged. See the protocol amendment for details.

For all completed steps, including steps from incomplete tasks, run:

```bash
python experiments/summarize_steps.py --root /workspace/calibration_page32_20261113
python experiments/summarize_steps.py --root /workspace/calibration_page64_20261112
```

This read-only CPU summary reports both raw and corrected rankings without any
subset search. It also lists every outcome changed by ID normalization.
