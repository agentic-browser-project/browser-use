# Independent review of real-KV intervention and selection protocols

2026-09-15. Read-only review; no experiment code or root report was changed.

Reviewed source hashes:

```text
real_kv_interventions.py  52b00c818adb6c6b7df94672154ad74cfbb3d9cda21d8c1dc5f30406b6455e16
selection_controls.py    8bbcccf2410958a671fff37f466f23c242620d588814350861dd7244bfb75ece
```

**Verdict: the core mathematics and array logic are correct for the stated diagnostic protocols.** The source/data audit supplies the necessary premise that captured text positions use ordinary split-half RoPE with θ base 5,000,000 and consecutive token-index differences. The new experiments support local, conditional conclusions about captured representations, not a model retrained or rerun without RoPE.

## Real-KV intervention script

For a window beginning at s and key at relative index j, the implemented transform is `R[(γ−1)jθ] K[s+j]`. Given `K[s+j]=R[(s+j)θ]X[s+j]`, this equals `R[(s+γj)θ]X[s+j]`. Thus γ=0 holds the first key's phase fixed and removes explicit phase differences within the window; γ=.25 and 4 slow/speed those differences. The content vectors at the captured layer stay fixed. Band masks restrict the operation to the intended rotary pairs.

The query is fixed across lengths and these interventions. Common shifts intentionally rotate both Q and K; their dot products are mathematically invariant. Split-half dimensions `(j,j+64)`, grouped query pooling, `/√128` scaling, per-head maxima, and support-function bounds are implemented consistently. `max` occurs before averaging heads, which is correctly documented. Extrema are recomputed from transformed keys. The code checks valid bounds and full-head logit invariance.

Matched starts, nested lengths, common seeds across layers, and exact file fingerprints improve reproducibility. Interpretation should preserve the following limits:

- Removal of relative phase changes true logits and true maxima. Width changes are not automatically changes in useless slack; report gap and maximum alongside width.
- The frozen layer content already includes positional effects from previous layers and learned adaptation. This is a local representation intervention, not an end-to-end no-RoPE model.
- Starts are sampled only where a 1024-token window fits. Shorter-length results intentionally exclude suffix starts that cannot support L=1024. This matches the position distribution across lengths but defines a restricted-start estimand.
- Three queries (first, fifth, last) are an equally weighted sparse-trajectory sample; they are not representative of every decode state by construction. They remain fixed within each comparison.
- CSV rows average windows and queries before saving. Seeds/windows overlap, repeated prompts exist, and layers share requests. Inference should cluster at task or capture/prompt level; seeds and layers are not independent prompt replicates.
- `band_*` measures unweighted pair half-range, while `pooled_width` uses absolute pooled-query weights. A band effect on one is not automatically the same effect on the other.
- Rotations act on FP16-captured values converted to FP32. Removing phase recovers an approximate frozen representation, not the exact original pre-RoPE tensor before quantization. Common-shift error checks quantify arithmetic invariance, not recovery of lost precision.
- The manifest is rewritten progressively. Downstream consumers must wait for a completed manifest containing every capture and non-null completion timing.

No blocking mathematical issue was found.

## Offline selection script

The logits have shape `[query, KV head, query-in-group, token]`. Flattening the middle two dimensions yields 32 query heads; applying softmax over tokens **before** averaging heads correctly computes mean full-head attention probabilities. Chunk masses sum these token probabilities. The score `c+w` uses the mean of eight query heads for each KV head and then averages KV-head bounds, matching the specified grouped-query proxy. The calibrated score is `c+(16/L)^.25 w`.

Both partitions cover every prefill token exactly once. The greedy rule sorts scores descending, skips chunks that do not fit, and admits complete chunks within budget 2048. It is correctly described as a diagnostic with no pinned tokens, system floors, or online generation. Common rotations are shared by all keys and query heads, and the code checks both logit and mass invariance. Token-weighted Jaccard is computed correctly because comparisons use the same disjoint partition.

Interpretation limits:

- `exact_density_greedy` uses exact mass density as a sorting score. It is not an exact solution of the indivisible 0–1 knapsack problem.
- Fixed16 includes a possibly shorter final chunk. Quest and width calibration can differ on that tail. Correlation with length in this partition is either undefined or driven entirely by a single short tail; it is not a broad measure of length bias.
- Chunk-relative collapse produces a different altered K for each partition, since each partition supplies its own origin per chunk. Native-versus-fixed16 collapse comparisons are partition-specific interventions, not scoring of a common counterfactual key cache.
- Collapse changes the dense target distribution too. Better retained mass under collapse does not show improved recovery of the **original** attention distribution or online action quality.
- Common-shift comparisons are stronger for axis sensitivity: every true attention probability stays fixed while the box representation and selections can change.
- Near ties, floating-point differences can change `exact_density_greedy` ranks after a common shift despite mathematical invariance. Its Jaccard provides a numerical baseline. Alternatively, after checking invariance, reusing original masses for common-shift density sorting/evaluation would eliminate that irrelevant numerical variation.
- The captured queries come from sparse decoding, and the softmax denominator covers only saved prefill keys. There are no saved values and no generated-token K in this calculation. Retained mass is a diagnostic, not a dense-trajectory result or output-error metric.
- Native chunks are the archived partition. This does not reproduce the original report's later parser-generated variable chunks, unless independently verified equal.
- `calls=[0,4,last]` presumes at least five saved query calls. The inspected dataset audit verifies that premise for all 23 captures.

No blocking arithmetic or array-shape issue was found. The data's duplicate prompts and repeated tasks should remain grouped during result aggregation.

## Geometry tutorial figure

`draw_rope_geometry_tutorial.py` creates `rope_geometry_tutorial.png` and `.svg` in this directory. Panel (a) shows one unchanged content vector rotated by θ=π/4 at positions 0,1,2. Panels (b) and (c) use exactly matched keys and query under a common 45° rotation. Assertions verify true logits are unchanged, original bound is 2, and rotated bound is 0. The final PNG was rendered and visually inspected; labels, axes, and numerical captions are legible and do not overlap.
