# RoPE / Quest source and data audit

Audit date: 2026-09-15. Read-only audit of local report and the report-named spark00 host. No original scripts, remote sources, services, paper, or Overleaf content were changed. No experiment or model forward was launched by this audit.

## Source hierarchy and original claim

Primary narrative: `/Users/shiqihe/Desktop/3_Project/browser-use/simulator/reports/2026-09-11_anchor_exponent_study.zh.md`, especially lines 119–164 (§2.3–2.4), 251–254 (§3.1), 569–601 (§7.6–7.7), and §6 reproducibility paths. Repository `AGENTS.md` guidelines before `<browser_use_docs>` were read first.

The report's explicit mechanism claim is that key-envelope width grows through content extreme values plus RoPE rotation in middle-frequency coordinate pairs. It reports endpoint growth exponents of 0.199 for pairs 0–7, 0.328 for 8–15, 0.289 for 16–23, and 0.248 for the total unweighted range; it describes the middle two bands as crossing about one radian across lengths 8–256. It then attributes β≈0.25 to key geometry and, after a five-model comparison, calls α=0.25 a constant of the scoring method (§7.7).

What was actually measured: range of already-rotated real keys in random contiguous spans, grouped by coordinate-pair index; separately, a query-weighted envelope width versus span length; separately, online sweeps of a scoring parameter. None of the original inspected scripts intervenes on RoPE while holding the same keys/content fixed. The original evidence is an association between frequency-band identity and width-growth rate, plus limited transfer measurements. It does not isolate the causal contribution of RoPE, establish that RoPE worsens Quest bound slack, or show a universal exponent.

The script docstring describes its test as “Prediction: mid-frequency bands carry the growth.” This prediction and the band ordering are consistent, but the causal explanation is stronger than the original design.

## Working data access

Host: `spark00.eecs.umich.edu` via the existing SSH configuration, user `shiqihe`. Container: `tsa_spark00` (`18e6838b7cc7` at audit time).

```sh
ssh -o BatchMode=yes spark00.eecs.umich.edu \
  'docker exec tsa_spark00 sh -lc "ls /workspace/scoring_case_study"'
```

Runtime: `/workspace/sglenv/bin/python` (Python 3.12 environment, transformers 4.57.1 in the historical beta logs). The host/container lacks `rg`; `grep` was used there. Known runtime library setting from `run_beta_one.sh`:

```sh
LD_LIBRARY_PATH=/workspace/sglenv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib64
```

Read-only resource snapshot at approximately 16:03 UTC: NVIDIA GB10, GPU utilization 0%, no GPU processes; host 119 GiB total memory, 116 GiB available, about 3.0 TiB free disk. This is a snapshot, not a reservation; check again before starting jobs.

| Container path | Verified contents |
|---|---|
| `/workspace/scoring_case_study/dumps_discord2048` | 9.1 GiB; real requests `req_001`–`req_013` |
| `/workspace/scoring_case_study/dumps_discord10` | 6.9 GiB; real requests `req_001`–`req_010` |
| `/workspace/sparse3way/offline_discord2048.jsonl` | 13 source rows, ordered mapping to request number +1 |
| `/workspace/sparse3way/offline_discord10.jsonl` | 10 source rows, ordered mapping to request number +1 |
| `/workspace/scoring_case_study/run_dump_leg.sh` | Capture source: batch size 1, concurrency 1, CUDA graph disabled; explicitly documents request ordering and `req_000` health probe |
| `/workspace/scoring_case_study/serve_dump13_m16_wn25_B2048_p16.log` | Historical capture log; confirms model, 48 layers, Hq32/Hkv4, head128, rope_base5000000, Python fallback for norm/RoPE |
| `/workspace/TreeSparseAttention/python/tree_sparse_selector.py.bak_preresel` | Surviving historical dump instrumentation; current selector no longer contains the same dump implementation |
| `/models/Qwen3-VL-30B-A3B-Instruct` | Model checkpoint, tokenizer, config available |

Both datasets also contain `req_000`, a 9-token health probe. Exclude this from all real-data experiments.

## Capture format and sampling units

Every real request has all 48 `k_layer_00.pt` through `k_layer_47.pt` files. Historical instrumentation saves the logical request's contiguous prefill K rows as CPU float16, shape `[T, 4, 128]`. These are post-normalization, post-RoPE keys obtained from the HF prefill KV cache. Historical instrumentation calls `_dump_prefill_state(k_rows, kv_indices, layer_id)`, so it uses the corrected logical K-row view of the interleaved cache, not the old incorrect flattened-buffer indexing discussed in the August report.

Every `q_layer_XX.bin` is raw float16. Load as `np.fromfile(path, dtype=np.float16).reshape(-1, 32, 128)`. One row is one selector call at that layer. All 48 layer files within each request have identical call counts (checked from file sizes). The dump stores the full query heads; pooling is performed later by the analysis script.

`topk.jsonl` uses `{c: call_number, l: layer_number, sel: chunk_id_list}`. `token_ids.json` contains exactly the prefill tokens. `chunks.json` has `seq_len`, `page_size`, original chunk spans and labels, and configuration metadata. **The audited discord files do not contain `user_start`, despite the later report table suggesting they do.** Recompute it from token IDs as the anchor analysis scripts do. No pre-RoPE keys, explicit key/query position IDs, values, or decode-token IDs are included in this dump format.

The capture queries come from a sparse decode trajectory. Original offline mass calculations apply softmax to these recorded queries against only the fixed prefill K, independently for all 32 heads and then average heads. This is a dense-attention calculation conditional on the captured sparse trajectory and prefill keys; it is not a fresh dense rollout and excludes newly generated decode K.

There are 23 real requests but **21 unique token-ID prompts**. Duplicates are:

- `dumps_discord2048/req_009` and `dumps_discord10/req_006` (ESPN--33, step_008).
- `dumps_discord2048/req_010` and `dumps_discord10/req_008` (ESPN--33, step_011).

They have different sparse decode trajectories / query call counts. Do not treat them as independent prompts. Multiple distinct steps also share tasks; group bootstrap or uncertainty at task level when feasible.

| Dataset | Request | Task | Step | T | Query calls per layer |
|---|---|---|---|---:|---:|
| discord2048 | 001 | Allrecipes--25 | 014 | 8482 | 301 |
| discord2048 | 002 | ArXiv--41 | 010 | 7958 | 1023 |
| discord2048 | 003 | ArXiv--41 | 013 | 8328 | 289 |
| discord2048 | 004 | BBC News--41 | 004 | 8750 | 301 |
| discord2048 | 005 | Cambridge Dictionary--26 | 008 | 7566 | 364 |
| discord2048 | 006 | Coursera--8 | 001 | 10165 | 205 |
| discord2048 | 007 | Coursera--8 | 012 | 15807 | 390 |
| discord2048 | 008 | ESPN--33 | 002 | 10292 | 285 |
| discord2048 | 009 | ESPN--33 | 008 | 15641 | 328 |
| discord2048 | 010 | ESPN--33 | 011 | 23143 | 421 |
| discord2048 | 011 | level1-6 | 002 | 8699 | 346 |
| discord2048 | 012 | Google Map--9 | 004 | 13840 | 1023 |
| discord2048 | 013 | Huggingface--22 | 014 | 13983 | 305 |
| discord10 | 001 | Allrecipes--25 | 007 | 7550 | 1023 |
| discord10 | 002 | Allrecipes--25 | 011 | 8068 | 314 |
| discord10 | 003 | Allrecipes--3 | 002 | 7360 | 305 |
| discord10 | 004 | Apple--0 | 002 | 6760 | 207 |
| discord10 | 005 | Coursera--8 | 010 | 12254 | 345 |
| discord10 | 006 | ESPN--33 | 008 | 15641 | 393 |
| discord10 | 007 | ESPN--33 | 010 | 20606 | 390 |
| discord10 | 008 | ESPN--33 | 011 | 23143 | 341 |
| discord10 | 009 | level1-5 | 002 | 8758 | 284 |
| discord10 | 010 | GitHub--22 | 001 | 6317 | 283 |

Original source row step strings have the form `step_014`; leading dataset path components above are shortened only in this table.

## RoPE configuration and position semantics

The local-on-host checkpoint's `config.json` directly verifies:

```json
{
  "head_dim": 128,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "num_hidden_layers": 48,
  "rope_theta": 5000000,
  "rope_scaling": {
    "mrope_interleaved": true,
    "mrope_section": [24, 20, 20],
    "rope_type": "default"
  }
}
```

Implementation inspected: `/workspace/sglenv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py`.

- `rotate_half`, lines 155–159, splits the head in halves and returns `(-x2,x1)`. The rotary pair is **(j,j+64), not adjacent dimensions (2j,2j+1)**.
- Attention forward, lines 272–281, applies per-head q/k RMS normalization, then rotation, then caches keys.
- `Qwen3VLMoeTextRotaryEmbedding`, lines 809–869, uses the default RoPE inverse-frequency initializer. For j=0..63, `theta_j = 5000000 ** (-2*j/128)`.
- Interleaved MRoPE maps the H phase to indices `1:60:3`, W to `2:60:3`, and leaves the other indices on T. This interleaving selects the position axis; it does not change the split-half pair layout.
- Pure-text `get_rope_index`, lines 1180–1195, uses attention-mask cumulative positions minus one, or `arange(T)` without a mask, and replicates the result identically across T/H/W. With identical axes, the interleaving yields ordinary 1D split-half RoPE.

Text-only status was checked independently in the data: all 23 input rows have string message contents; all 23 `token_ids.json` files have zero image token 151655 and video token 151656. Capture runs use one request at a time. The source path prepares text-only chat input and HF prefill; the historical log confirms ordinary 5000000 base even though mRoPE uses its Python fallback. Thus key-position differences within a contiguous prefill span are the token-index differences. Explicit historical position tensors are not saved, so absolute positions are inferred from the verified text path; **relative position i−s is sufficient for the planned anchor-relative intervention**.

The current serving source decodes its first generated token from prefill logits, then uses position `T+i−1` for loop iteration i≥1; this is consistent with selector call c=0 at position T. The new intervention does not require relying on this absolute-query-position reconstruction.

## Exact original analysis protocols and discrepancies

### `rope_band_width.py`

- Layers `[0,12,24,36]`, span lengths `[8,16,32,64,128,256]`, 120 spans per length/layer/prompt, NumPy RNG seed 0.
- It constructs the 13 discord2048 paths followed by 10 discord10 paths, filters existing token files, then takes `[:12]`. Therefore it measures **only discord2048 req001–012**.
- For each length it draws a new set of starts, independently across lengths. Starts use `rng.integers(0, T-L)`, excluding the final valid start `T-L` (minor off-by-one sampling omission).
- Each span computes `r = max_i K_i - min_i K_i`, then pair range `0.5*(r[:,:64]+r[:,64:])`, averages KV heads, and finally bands of eight pairs. This is **unweighted key range**, not the query-weighted w and not Quest gap.
- Band β is the endpoint slope `log(range256/range8)/log32`; total β is likewise endpoint-based. This differs from six-point least-squares β used by other scripts.
- The audited script contains `theta = 1e6 ** (-2*np.arange(64)/128)` and prints 1e6 in its header. **This only affects printed labels/angle descriptions, not the measured ranges or bands.** The report's 5e6 frequency labels agree with the actual checkpoint; the script labels are stale. A corresponding rope-band output log was not found in the scanned `analysis/*.log` or top-level `*.log` files, so the reported band numbers were not independently matched to a saved raw output by this audit.

### `measure_width.py`

- Layers `[0,12,24,36]`; same six lengths; 150 spans per length; one query per layer, specifically `qall[min(4,nq-1)]`, i.e. the fifth selector call for all real requests.
- Query pooling is `q.view(4,8,128).mean(1).abs()`. It computes `mean_h sum_d abs(mean_g q_gd)*(M−m)/2`, without the 1/sqrt128 scale.
- The audited code uses `sorted(os.listdir(dataset))[:3]` for each set. These are warmup req000 plus real req001 and req002. Warmup T=9 is skipped for every tested length by `T <= L+1`. Therefore the saved implementation has **4 contributing real prompts × 4 layers**, not the stated 23 prompts × 4 layers.
- `analysis/measure_width.log` exactly matches the report's mean width table (116.102,151.314,185.590,219.197,250.548,281.127), pooled β0.245, min0.192/max0.279. Without a historical copy/log of an alternate script, the narrative's 23-prompt attribution is unsupported; the matching saved script and output point to 4 prompts.
- This β is a least-squares fit across the six log-lengths, not just the endpoint slope. Each length uses independent sampled starts.

### `anchor_insight.py` / `anchor_insight2.py`

- They iterate all 13+10 real rows and use the same four layers. Calls are `range(0,nq,32)` plus the last four calls, with duplicates removed.
- Region chunks are recomputed from token decoding / parser, using DOM min6/max64 and other min64/max256, rather than blindly reusing historical capture chunks.
- Grouped q means eight query heads per KV head; final scores average four KV heads. Center/width and grouped-q true maxima are scaled by 1/sqrt128. Full-head mass uses all 32 heads separately before averaging softmax mass.
- Consequently `c+w` bounds the grouped-query logits (and the corresponding mean-of-head-max quantity), **not each original query head's logit**. The ordering or bound interpretation should not silently switch between pooled and full-head queries.
- Reported correlations and grouped summaries average per-query statistics; longer decodes supply more sampled queries. Score/mass ties are ranked with stable ordinal `argsort` rather than average ranks.
- Saved `analysis/anchor_insight{,2}.{log,json}` are present. None is a RoPE ablation.

### `beta_hf.py`

- Actual code defaults to 6 text-only strat20 prompts, max8000 tokens, four quarter-depth layers, 150 spans/length. Its docstring is stale (says 8 prompts and 12k).
- It performs a fresh HF model forward and measures unweighted post-RoPE K range. Four non-30B models and two 30B datasets have saved logs / JSON in `analysis/beta_*.{log,json}`.
- The 30B WebVoyager HF log gives β0.246 (six-point least squares), whereas the original twelve-dump band's β0.248 is an endpoint slope on different data. These are compatible numerically but are not the same estimand/protocol.
- Different learned models, RoPE bases, head counts, and statistics change together. Their similar β does not causally establish insensitivity to base or a universal α.

## Causal controls needed and interpretation of new experiments

The original range-by-band observation alone cannot distinguish RoPE phase spread from coordinate-specific learned content, q/k scale, norms, temporal correlation, domain structure, or the effect of different random windows. The statement that small θL implies range linear in L requires locally fixed/smooth content; it is not true for arbitrary changing key content. Likewise, “high-frequency band saturated at L=8” is too strong for all j=0..7: at j=7, θ≈0.185, so eight tokens span only about 1.3 radians, not a full circle.

The planned controlled experiment can use existing real K without knowing absolute query position:

`K'_i = R_{(gamma-1)*(i-s)*theta} K_i` for a window starting at s.

At gamma=0 this removes explicit within-window RoPE phase spread while preserving K_s; gamma=1 is the original; other gamma values change phase speed at fixed captured content. Use the same recorded q, same window starts, and nested lengths. Recompute both bounds and true maxima for every condition, since changed K changes logits and their true maximum. Run band-specific versions to distinguish bands' causal effects on this frozen-key statistic.

A common per-pair rotation applied to **both** every K and q preserves all pairwise dot products and true maxima, while axis-aligned boxes generally change. It is a useful control specifically for coordinate-frame sensitivity of the box bound. Verify logit invariance numerically. This is distinct from removing relative positional spread.

Even a successful dephasing effect is a causal statement about an intervention on **captured representations**, not a claim about what a model retrained without RoPE would do. Earlier-layer positional influences remain encoded in the content portion of deeper-layer K. Dephasing can also move tensors away from the learned representation distribution. Report β, width, true max, and gap separately; lower width does not automatically mean a tighter bound relative to its changed true maximum or better token selection.

Statistical controls: pair windows and queries across conditions; report layer effects instead of only a pooled average; preserve separate pooled-query and full-head-query results; separate twelve-prompt original-protocol replication from expanded-data results; account for task grouping and exact-prompt duplicates.

## Fingerprints for provenance

SHA-256 at audit time (remote original sources, unmodified):

```text
4671da0e518431622b09e90c53e6b1d338112a6a3d3c7815c5bb780a8c5d51bf  rope_band_width.py
b01c14211022eb7428c1bf194b0f7c932ed3835e015b640e3f9e2cf060c0d19c  measure_width.py
5ad4e3f65798d870c2d2e05052536e54c6ed69b2c1905dab8f461256d0f0cbdf  anchor_insight.py
333a14826957400534856a0b8c3795795eb7b5db9f6536cf25fda70cda0560e7  anchor_insight2.py
317d03ec418e1c91574eec86bd311710efad8fe4fe766cdf402dbe7865388543  beta_hf.py
6129b3f54d517a18553cf7525ff4886be6831c5806fc094d632b8ef38d859912  /models/Qwen3-VL-30B-A3B-Instruct/config.json
70329972cf44fae227710991f3660233cfa3563cd5a5f2e637ecef4794acb2c8  dumps_discord2048/req_001/k_layer_00.pt
c1e21678daa6776d3a16b69c5906c1ca57f44518d25ddc57149c861e429b259e  dumps_discord10/req_001/k_layer_00.pt
```

Paths without absolute prefixes in this fingerprint block are under `/workspace/scoring_case_study/`.
