# Revised offline calibration protocol

The user confirmed this revision on 2026-11-12. It supersedes the two-profile grid and early stopping rule of the preceding calibration32 experiment.

## Inference and input sample

Use Qwen3-32B BF16 on spark00 only, with greedy generation, thinking enabled, and at most 1024 output tokens. Each of the 64 distinct WebVoyager tasks contributes one archived step. The task list, step selection, ordering, text messages, and tokenization are unchanged from the previously frozen candidate manifest. These are indexed-action source steps, not necessarily steps from successful tasks. No browser action is executed.

The only profile uses 64-token physical pages, maximum chunk size 32, and soft merge target 6. All prompt tokens remain represented, including boundary chunks shorter than 6. The nominal distinct-page budget is 12.5%, with the existing mandatory reservations unchanged.

Measure all 24 combinations of A in {1,2,4,8,16,32,64,128} and alpha in {0.125,0.25,0.5}, plus one dense reference. Sparse variants share dense prefill but generate their own subsequent hidden states and KV caches. Fixed pairs of prompts yield 50 decoder rows. The preceding two completed tasks are preserved separately and rerun under this schedule, since BF16 free generation was observed to depend on batch membership.

## Validation

Use the existing pinned BrowSA chunk parser, pooling, packing, and page selector. Compare the optimized KV gather against an independent direct-index reference bit for bit. Compare optimized and reference full-model replay under identical new batch membership, row order, and cache capacity before measuring new tasks. Controls use the first 32 generated tokens and do not claim exhaustive equivalence for every long response. Reuse prompt KV captures only after matching the exact input-token hash.

## Rolling subset search

Start analysis after 16 completed steps and repeat after every completed pair through all 64 tasks. Finding a favorable subset does not stop measurement. Search separately for sets of exactly 16 and exactly 32 distinct tasks once enough dense-index-eligible states exist. The 16-task search is an early exploratory result, and the 32-task search preserves the earlier dataset-size objective.

The primary criterion is exact agreement with the dense response in the entire action-type sequence and flattened target-index sequence. It does not require matching other arguments or establish that the dense action is correct. Invalid or unfinished sparse responses count as mismatches. States without a completed, parseable dense indexed action are ineligible for this metric, remain in all exports, and are never silently replaced.

For each subset size, maximize the minimum match-count margin of (32,0.25) against every other configuration. Break ties by maximizing its match count. Positive margin means uniquely first, zero means first including ties, and a negative exact optimum means no first-place subset of that size exists in the measured eligible pool. Preserve each rolling snapshot. Also report full action equality, action-type agreement, fixed primary cohorts, and the entire measured pool.

The search is explicitly outcome-dependent and favors a named configuration. Selected subsets answer an existence question within this pool. Their ranks are not unbiased evidence of general superiority or online task accuracy. Do not change the metric or exclude unfavorable outcomes to manufacture a win.

## Metric correction on 2026-11-13

Integer target IDs and equivalent integer-string encodings now match, for example
`8117` and `"8117"`. This correction applies uniformly to every configuration in
both page profiles. Target order and multiplicity, action-type sequences, dense
eligibility, and the finished/parseable-output requirements are unchanged. XML
and alternate target-field names are not newly accepted. Booleans and floats
are not coerced to integer IDs. No inference outputs are modified or regenerated.

`target_equal` uses the corrected comparison. `target_equal_raw` retains the old
Python list comparison for audit. Analysis manifests identify the metric as
`integer_target_ids_v2`. Earlier results remain historical raw-metric results,
not estimates under the corrected rule.
