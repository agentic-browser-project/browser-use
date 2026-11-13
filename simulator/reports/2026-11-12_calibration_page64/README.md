# Page64 / maximum-chunk32 replay

This is the previous Qwen3-32B BF16 experiment, stopped at the user's request after
56 WebVoyager steps from 56 distinct tasks. It is not the currently running job.

- [`experiments/run_replay.py`](experiments/run_replay.py) is the inference entry point.
- [`experiments/page64_attention.py`](experiments/page64_attention.py) implements sparse decoding.
- [`experiments/fused_kv_gather.cu`](experiments/fused_kv_gather.cu) implements fused page gathering.
- [`settings.json`](settings.json) contains the 24-setting A × alpha grid,
  page size 64, maximum chunk size 32, and nominal 12.5% page budget.
- [`experiments/analyze_replay.py`](experiments/analyze_replay.py) retains the full
  measured pool separately from the outcome-selected subsets.
- [`runtime_manifest.json`](runtime_manifest.json) identifies the executed code.

Each task has one step here, so task-weighted and step-weighted rankings agree.
All 56 completed steps are retained. Forty have a finished dense reference with
parseable indexed actions and enter the action-type plus target-index metric.
Ten dense responses did not finish and six had no target index. Sparse failures
remain mismatches in the common 40-step denominator.

The shared runtime dependencies and environment are documented in the
[current page32 experiment README](../2026-11-13_calibration_page32/README.md#code-dependencies-and-execution-paths).
Inference requires the external model, frozen inputs, and pinned BrowSA source
snapshot. These large runtime artifacts are not part of this code upload.
See [`protocol.md`](protocol.md) for experiment and subset-search definitions.
