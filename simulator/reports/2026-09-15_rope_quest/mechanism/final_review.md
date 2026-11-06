# Final independent review

2026-09-15. Scope: mathematical and causal claims in `report.md`, real/selection CSV results and aggregation, beginner explanation, and all three real-data figures. The literature attribution review is handled separately; this review did not re-fetch primary papers. The report and experiment scripts were not edited by this reviewer.

## Verification performed

`verify_final_report.py` independently re-aggregates the raw CSV files without importing either analysis script. It averages capture cells, duplicate captures within prompt, prompts within task, and the 13 tasks equally. It also independently recomputes the key paired bootstrap intervals using the declared 5,000 task resamples and seed.

Results recorded in `final_verification.json`:

- 30,360 real-intervention rows and 8,280 selection rows checked.
- All **2,013 curve estimates** reproduced exactly.
- All **150 selection estimates** reproduced to within `2.3e−13`.
- Headline percentile intervals independently reproduced for width reduction, gap reduction, slope change, calibration gain, and shift-4093 mass change.
- All **1,656 exact-density common-rotation selections** are unchanged.
- Maximum window logit discrepancy is `1.90735e−5`; maximum selection chunk-mass discrepancy is `3.57628e−7`.

The quoted widths, gaps, true maxima, per-head reductions, endpoint-band exponents, local exponents, bound changes, selection masses, budgets, chunk sizes, and Jaccard values agree with the saved CSVs to the stated precision, apart from the minor slope-CI rounding note below.

## Mathematical and causal assessment

The RoPE relative-position identity, split-half pairing, positional span `(L−1)θ`, box support-function identity, width/gap distinction, Gaussian extrema calculation, identical-content sufficiency control, isotropic Gaussian null, and common-rotation invariance are correct.

The report correctly distinguishes a **local operation on frozen representations** from removing RoPE throughout inference or retraining. It explicitly preserves upstream positional effects as an unidentified part of the residual. It does not assign the residual to position-free content, infer downstream success from mass, or treat the 8.65% local gap decrease as an attribution of all sparse-decoding errors.

The statistics use task clustering and exact-prompt deduplication. Paired changes are computed from joint resamples, rather than subtracting independent marginal confidence limits. The report now states that intervals are pointwise and lack multiplicity correction. Their tightness must remain a property of these conditional measurements, not evidence of coverage across unseen domains or checkpoints.

The strongest selection evidence is correctly identified: a common Q/K rotation preserves the attention target while changing Quest's selections. The exact-density stability is a useful numerical control. The report does not use collapse-selection mass to claim improvement for the original target and distinguishes archived native boundaries from reconstructed report chunking.

The beginner progression is coherent: vectors and attention → pair rotations → frequency/span → bounding boxes → true maximum versus gap → budget and mass. The three-panel geometry tutorial uses genuinely matched Q/K in its common-rotation comparison.

## Figure inspection

- `real_width_and_gap.png`: width and gap are in separate panels, scale units are explicit, logarithmic length ticks are clear, the 256-token gap annotation agrees with the data, and both residual-growth curves remain visible.
- `real_frequency_interventions.png`: endpoint exponents are labeled distinctly from OLS width slopes; legends, pair bands, speed interventions, and error bars are legible. The fourfold peak shift and quarter-speed high-band increase are visible.
- `selection_controls.png`: retained prefill mass is clearly marked as a percentage; fixed16 includes its tail label; the shared-rotation panel labels Jaccard and shows the invariant exact-density control. No clipping, overlapping labels, or mismatched values were found.

The three tested shifts are displayed at categorical positions. They should be understood as three intervention settings, not a fitted continuous relationship between numerical shift magnitude and Jaccard.

## Concrete corrections sent to the report owner

The owner has already applied these during review:

1. Add the required `--input`, `--manifest`, and `--output` arguments to the `analyze_selection.py` reproducibility command.
2. Replace “middle pairs explain most of the reduction” with “produce the largest reduction among the band-specific interventions.” This reports the actual intervention comparison without treating nonlinear slopes as additive causal shares.
3. State that reported intervals are pointwise and uncorrected for the multiple metrics/interventions examined.

The owner also applied these final precision refinements, verified in the updated report:

4. The lower paired slope-change CI is `−0.012848688...`; at four decimals it rounds to **−0.0128**, not −0.0129. Five-decimal reporting would give −0.01285.
5. “Density greedy is exactly optimal only when chunks are fractional” is unnecessarily exclusive: equal-cost whole chunks are also an exact special case. Prefer: “Density greedy has a general optimality guarantee for the fractional relaxation; for indivisible variable-length chunks it is a heuristic.”
6. In the opening summary, “changing their speed” could imply changing only middle-band frequencies. The actual speed interventions change all pair frequencies together; “changing rotation speed” is precise.
7. A heading such as “Removing within-window phase leaves substantial growth” avoids shorthand that could momentarily suggest a full no-RoPE model. The body already makes the intended scope clear.

The revised §5.2 heading is “Measured contribution and remaining growth.” The other requested refinements are present as described above.

**Final assessment: satisfied.** Experimental mathematics, raw-to-summary numerical verification, substantive causal boundaries, beginner explanation, and figures pass this review. All requested corrections have been applied; no unresolved issue remains within this review's scope.
