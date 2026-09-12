# Sparse Attention for a Web Agent: TSA vs Quest vs BlockSparse

**A complete write-up, assuming no background.**

Model under test: `Qwen3-VL-30B-A3B-Instruct`. Hardware: NVIDIA GB10 (sm_121), 119 GB unified memory.
Workload: WebVoyager + GAIA browser-agent trajectories.
Dates: 2026-08-03 → 2026-08-07. All numbers in this document were measured in that window unless a
citation says otherwise.

---

## 0. TL;DR

1. **Sparse attention costs accuracy on this workload, and buys no measurable speed.**
2. **The reason it buys no speed is not a bug in the sparse code — it is that attention is only ~8.5%
   of a decode step on this model at these context lengths.** The theoretical best case from perfect
   sparsity is a **1.09x** end-to-end speedup. That is smaller than the measurement noise.
3. **The actual bottleneck is a Mixture-of-Experts kernel chosen by a silent HuggingFace default.**
   `experts_implementation="grouped_mm"` costs **77.1 ms of the 153.5 ms decode step** — half of it —
   and runs **4.7x slower than its own memory-bandwidth roofline**. In a microbenchmark, switching
   one environment variable to `batched_mm` saves ~33 ms. *One flag beats the entire algorithm* —
   **but that win has not yet been confirmed end-to-end** (§8.6).
4. **Batch size is the strongest lever on attention's share (§9.3)** — 32% at B=1, 64% at B=32, above the 55% Quest reports for Llama-2-7B. This report previously said "batch is not a lever"; that was wrong. KV traffic is exactly linear in B while weight traffic is sublinear and saturates. **Every speed experiment here ran at B=1 — the setting least favourable to the method.**
5. **On accuracy, budget is what matters, and the ranking flips with budget.** At a 4096-token budget
   TSA is significantly better than both baselines; at 8192 the two baselines become statistically
   indistinguishable from full attention while TSA is still significantly behind.
6. **Several of my own earlier measurements were wrong and are retracted here** (§5). They failed
   silently — producing plausible numbers rather than errors — which is why §5 exists.

---

## 1. What problem is this trying to solve?

### 1.1 The web agent

A "web agent" is an LLM driving a real browser. Each step, the agent receives a text description of
the current page — every clickable element, each tagged with a numeric **element index** — and must
reply with an action such as `click(index=41827)`.

Those page descriptions are large. In this dataset they run **25,000 to 118,000 characters**
(~6,000 to ~29,500 tokens), and they accumulate across steps. Long context is not optional here;
it is the nature of the task.

### 1.2 Why long context is expensive

A transformer stores a **KV cache**: two vectors (a "key" and a "value") per token per layer, kept so
that later tokens can attend to earlier ones. To generate **one** token, the model reads the *entire*
KV cache. Twice the context, twice the reading.

### 1.3 What sparse attention proposes

Most of that KV cache is irrelevant to any given query. **Sparse attention** picks a small subset —
a *budget* — and reads only that. If you can pick well, you read 2,048 tokens instead of 29,500 and
the model behaves almost the same.

Two questions follow, and this project measures both:

- **Accuracy:** does the model still choose the right element index?
- **Speed:** does reading less KV actually make generation faster?

The answers turn out to be "no, it degrades" and "no, it doesn't" — and the *reason* for the second
answer is the most useful result in this report.

---

## 2. The three methods

All three split the KV cache into **chunks**, score each chunk against the current query, keep the
top-scoring chunks, and attend only to those. They differ in exactly two design choices.

### 2.1 Design choice A — how to cut the context into chunks

**Fixed chunking (Quest, BlockSparse).** Cut every 16 tokens. Simple, uniform, structure-blind.

**Variable/tree chunking (TSA — the method under development here).** Cut on *document structure*.
A browser page has natural boundaries — one element, one message, one section — and TSA's parser
emits chunks of 16 to 256 tokens aligned to them (`--tree-parse-mode webarena`).

*Why this could matter.* Scoring compresses a chunk to one summary vector. The bigger the chunk, the
more a single distinctive token gets diluted by its neighbours. Earlier in this project we measured
exactly that: chunks of 1–8 tokens were selected 84.4% of the time, chunks of 129+ tokens only 8.4%.
Structure-aligned chunks keep semantically-coherent things together, so the summary is less of a lie.

### 2.2 Design choice B — how to score a chunk

**Centroid scoring (BlockSparse).** Average the chunk's key vectors into one mean vector; score by
dot product with the query. *Failure mode:* averaging cancels. One key pointing hard at the query,
averaged with fifteen pointing elsewhere, produces a mediocre mean — and the chunk containing the
answer scores low.

Plain-language version: to decide whether a book is worth reading, you read the average of every
sentence in it. A book with one crucial sentence averages out to noise.

**Envelope scoring (Quest, TSA).** Keep the per-dimension **min and max** across the chunk, then
score the *upper bound* of what any key in that chunk could achieve: take the max where the query is
positive and the min where it is negative. This is an admissible over-estimate — it never
under-rates a chunk that contains a match.

Plain-language version: instead of the average, you ask "what is the *best* sentence in this book,
optimistically?" A book with one crucial sentence now scores high.

### 2.3 The three configurations

| Method | Chunking | Scoring | Launch flags |
|---|---|---|---|
| **TSA (tree)** | variable, structure-aligned | envelope | `--page-size 64 --tree-parse-mode webarena --scoring-method envelope` |
| **Quest** | fixed, 16 tok | envelope | `--page-size 16 --tree-parse-mode fixed --scoring-method envelope` |
| **BlockSparse** | fixed, 16 tok | centroid | `--page-size 16 --tree-parse-mode fixed --scoring-method centroid` |

TSA differs from Quest only in chunking; Quest differs from BlockSparse only in scoring. **The
comparison isolates one variable at a time.** That is the point of the design.

### 2.4 The fairness control: matched budget

`top_k` alone is meaningless across methods, because the page sizes differ 4x. What must be held
constant is **budget = page_size × top_k**, the number of KV tokens actually attended per decode step.

| Budget | TSA | Quest / BlockSparse |
|---|---|---|
| 2048 | page 64 × k 32 | page 16 × k 128 |
| 4096 | page 64 × k 64 | page 16 × k 256 |
| 8192 | page 64 × k 128 | page 16 × k 512 |

For scale: at 118k characters (~29.5k tokens), a 4096 budget reads about **1/7** of the context.

---

## 3. System implementation

### 3.1 The model

| Property | Value | Why it matters here |
|---|---|---|
| Layers | 48 | Every per-layer cost is paid 48x per token |
| Experts | 128, **8 active** per token | 30B total but only ~3B active — weights dominate anyway |
| Hidden size | 2048 | |
| Attention heads | 32 query / **4 KV** (GQA 8:1) | **KV cache is 8x smaller than MHA** — see §9 |
| Head dim | **128** | Not `hidden/heads = 64`. Getting this wrong halved my KV estimate (§5.4) |
| KV per token | 2 × 4 × 128 × 2 B × 48 layers = **96 KiB** | |
| Active weights read per step | **6.083 GB** | |
| dtype | bfloat16 | |

### 3.2 The serving stack

Not vLLM or SGLang — a purpose-built `serve.py` (1,251 lines) with FlashInfer paged KV. All three
methods run through **the same binary**; only launch flags differ. This is a strength (no
cross-framework confound) and a weakness (no framework-level optimisations to lean on).

Per decode step, per layer, the sparse path runs: group query heads → quantise query to fp8 → score
all chunks → fused page-select → top-k → build page indices. Roughly 250–290 extra kernel launches
per step.

### 3.3 Measured hardware

- Effective memory read bandwidth: **222.7 GB/s** (measured on a 512 MB bf16 reduction; spec 273)
- Unified memory: 119 GB total, shared between host and device

---

## 4. Experimental design

### 4.1 Data

100 tasks sampled from WebVoyager + GAIA, all of which a reference model (`Qwen3.5-Omni`, dense
attention) completed successfully, with ≥1 task per site category. Split into two halves of 50
**complete trajectories** — every step of every task, never a step sample, because a partial
trajectory cannot be evaluated end-to-end.

All results below use **half1: 50 tasks / 332 steps / 190 index-emitting steps / 142 other steps**.

### 4.2 Metrics

For each step we replay the exact reference context and compare the model's action:

- **`valid`** — the chosen element index actually exists on the page.
- **`agree`** — the chosen index matches the reference trajectory's index exactly.
- **`none`** — the model emitted no index at all.
- For non-index steps (scroll, done, …), whether the **action type** matches.

**`agree` is a strict proxy, not task success.** Many tasks have several valid paths; clicking a
different-but-correct button counts as disagreement. §7.2 quantifies exactly how strict, using the
full-attention run as the yardstick.

### 4.3 Statistics

Steps are **nested within tasks**, so treating 190 steps as independent overstates significance.
Every test below is therefore **cluster-level**, with the *task* as the unit: a sign test on which
config got more index steps right per task, plus a cluster bootstrap (resample tasks, recompute the
pooled agree-rate difference).

---

## 5. Methodology failures found and fixed

These are documented because every one of them produced **plausible numbers rather than an error**.
Three separate rounds of results were discarded.

### 5.1 `ignore_eos` did nothing (silent, cost: one full benchmark pass)

To compare per-token latency across configs, every config must generate the *same* number of tokens.
I added an `ignore_eos` flag and verified it: cells came back at exactly 128 tokens. **The
verification was a coincidence.** Browser prompts naturally produce long JSON, so nearly every cell
hit the `max_tokens` ceiling; `ignore_eos` was irrelevant.

Root cause: `serve.py` applies a **builtin JSON grammar whenever no schema is supplied**, and
decoding stops when `matcher.is_terminated()` fires as the JSON object closes. That break was not
gated by `ignore_eos`. A direct test with a short prompt exposed it: 21 tokens with the flag on, 21
with it off. Fixed by bypassing the grammar entirely in benchmark mode.

### 5.2 Two-call subtraction was destroyed by the scheduler (cost: one full pass)

Decode time was computed as `time(max_tokens=129) − time(max_tokens=1)`. Both calls queue behind the
same 150 ms batch-collect scheduler, so queueing noise lands in either term. Symptoms: four cells had
`decode_s ≤ 0` (reported as `2.56 × 10⁸ tok/s`), and — the decisive tell — **prefill for identical
work varied 20x across configs** (2.16 s vs 45.2 s) even though sparsity only applies at decode.

Fixed by sweeping `max_tokens` and fitting `time = intercept + slope × tokens`. The intercept is
prefill, which **must** agree across configs — a built-in validity check. After the fix, prefill
spread was **1.08x**.

### 5.3 The fixed estimator still could not resolve what it measured (cost: one full pass)

`--points 1,65,129` is symmetric, so the OLS slope is *identically* `(t₁₂₉ − t₁)/128`: **the middle
point has zero leverage** and R² measured only its agreement, not the slope's precision. With `n=1`
per cell that leaves a **±20 ms 95% CI on any config-vs-config difference** — wider than every
difference in the table. The reported per-config speed ranking was noise.

Fixed in `bench5.py`: five leveraged points, three repeats each, and the slope's **standard error**
is reported so differences can be tested rather than eyeballed.

### 5.4 `head_dim` assumed to be 64; it is 128

I computed `head_dim = hidden/num_heads = 2048/32 = 64`. The server log prints
`QO heads: 32, KV heads: 4, Head dim: 128`. Every KV-traffic number was **halved**. The claim
"attention is under 2% of decode time" was a consequence; the correct bound is **8.5%–15%**.

### 5.5 The "CUDA graph is unusable" verdict was a harness false negative

The probe declared death on a **single** `p=0` reading, with no debounce (unlike `wait_ready`, which
requires three consecutive). Combined with a `sleep 45` that was too short after `docker restart`,
all three backend attempts "died" in 51–52 s and two of the log files were **0 bytes** — no process
ever started. The configuration declared dead was later found **serving normally**. §8.3 has the real
answer.

### 5.6 `full` is not a dense baseline

`tree_sparse_selector.py:165` is `selected_chunks = min(top_k_chunks, total_chunks)`. With
`--top-k 100000` every chunk is selected — but the entire selection pipeline still runs. The `full`
arm is *sparse-code-with-everything-selected*, estimated to cost **+2 to +4 ms** versus a true dense
path. That is a fifth of the resolution, so it flips no conclusion, but it means **every ratio in the
speed campaign is sparse-vs-sparse** and the arm should be labelled `topk=all`.

### 5.7 `traj_eval.py` has no resume

It opens `partial_<tag>.jsonl` with mode `"w"`. Every "self-healing" relaunch silently restarted from
zero. Results stayed correct; hours were wasted.

---

## 6. Accuracy results

50 tasks / 332 steps / 190 index steps. **`DENSE` = full attention, the same replay, everything else
identical.**

| Config | Budget | `idx_valid` | `idx_agree` | non-index type agree | task all-agree | mean per-task fraction |
|---|---|---|---|---|---|---|
| **DENSE** | full | **154/190 (81.1%)** | **97/190 (51.1%)** | 62/142 (44%) | 16/50 | 0.620 |
| tree_k64 | 4096 | 134/190 (70.5%) | 55/190 (28.9%) | 61/142 (43%) | 6/50 | 0.364 |
| quest_B4096 | 4096 | 132/190 (69.5%) | 39/190 (20.5%) | 45/142 (32%) | 8/50 | 0.312 |
| block_B4096 | 4096 | 136/190 (71.6%) | 35/190 (18.4%) | 50/142 (35%) | 5/50 | 0.248 |
| tree_k128 | 8192 | 151/190 (79.5%) | 83/190 (43.7%) | 58/142 (41%) | 12/50 | 0.524 |
| quest_B8192 | 8192 | 139/190 (73.2%) | 90/190 (47.4%) | 68/142 (48%) | **16/50** | 0.583 |
| block_B8192 | 8192 | 151/190 (79.5%) | 91/190 (47.9%) | 59/142 (42%) | 15/50 | 0.594 |
| **tree_k256** | **16384** | 151/190 (79.5%) | **97/190 (51.1%)** | — | — | **0.624** |

**`tree_k256` reproduces full attention exactly** — 97/190 on both, mean per-task fraction 0.624 vs
0.620. See §6.1 and §7.1.

### 6.1 Distance from full attention

| Comparison | dense better on | worse on | p | agree-rate gap, 95% CI |
|---|---|---|---|---|
| vs tree_k64 (4096) | 25 tasks | 0 | <0.0001 ✱ | +0.152 … +0.296 |
| vs quest_B4096 | 32 | 1 | <0.0001 ✱ | +0.229 … +0.383 |
| vs block_B4096 | 33 | 1 | <0.0001 ✱ | +0.240 … +0.414 |
| vs tree_k128 (8192) | 14 | 2 | **0.0042** ✱ | +0.032 … +0.120 |
| vs quest_B8192 (8192) | 13 | 8 | 0.383 ns | −0.023 … **+0.095** |
| vs block_B8192 (8192) | 9 | 6 | 0.607 ns | −0.026 … **+0.094** |
| **vs tree_k256 (16384)** | **5** | **5** | **1.000 ns** | **−0.033 … +0.033** |

**Read the confidence intervals, not just the p-values.** All three "ns" rows fail to reject, but
they are not equally informative. Quest and BlockSparse at 8192 have intervals stretching to +9.5
points — that is *absence of evidence*, consistent with anything from parity to a substantial gap at
this sample size. `tree_k256`'s interval is ±3.3 points: **evidence of absence.** Only TSA at 16384
has actually been shown equivalent to full attention.

### 6.2 Within-budget comparisons

| Comparison | sign test | p | difference, 95% CI |
|---|---|---|---|
| **tree_k64 vs quest_B4096** | 19 / 6 | **0.015** ✱ | +0.030 … +0.140 |
| **tree_k64 vs block_B4096** | 26 / 11 | **0.020** ✱ | +0.033 … +0.179 |
| quest_B4096 vs block_B4096 | 12 / 7 | 0.359 ns | −0.029 … +0.070 |
| tree_k128 vs quest_B8192 | 10 / 14 | 0.541 ns | −0.107 … +0.027 |
| tree_k128 vs block_B8192 | 7 / 13 | 0.263 ns | −0.114 … +0.028 |
| quest_B8192 vs block_B8192 | 8 / 8 | 1.000 ns | −0.050 … +0.040 |

### 6.3 Failure-mode breakdown — the cleanest causal result in the report

| Config | agree | valid but different | **invalid (hallucinated index)** | none emitted |
|---|---|---|---|---|
| **DENSE** | 51% | 30% | **0%** | 19% |
| tree_k64 | 29% | 42% | 7% | 23% |
| quest_B4096 | 21% | 49% | 6% | 24% |
| block_B4096 | 18% | 53% | **10%** | 18% |
| tree_k128 | 44% | 36% | 3% | 18% |
| quest_B8192 | 47% | 26% | 6% | 21% |
| block_B8192 | 48% | 32% | **1%** | 19% |

**Full attention hallucinated an element index 0 times out of 190.** Every sparse config does it
1–10% of the time. Attribution is unambiguous: *inventing a page element that does not exist is
caused by sparsity.*

Equally important, **"emitted no index" sits at 18–24% for every config including dense.** That
failure is *not* caused by sparsity and must not be counted against it.

### 6.4 Multi-step tasks

Of 38 tasks with ≥2 index steps, how many had *every* index step correct:

| | DENSE | tree_k64 | quest_B4096 | block_B4096 | tree_k128 | quest_B8192 | block_B8192 |
|---|---|---|---|---|---|---|---|
| all agree | **5/38** | 0/38 | 0/38 | 0/38 | 2/38 | **6/38** | 4/38 |
| all valid | **21/38** | 12/38 | 14/38 | 16/38 | 16/38 | 16/38 | **21/38** |

Note dense itself only manages 5/38. **This metric is mostly measuring its own strictness.** With a
per-step agree rate of `p`, a `k`-step task needs `p^k`: at dense's 51%, a 5-step task is 3.5%.
Getting half of 5-step tasks fully right would require a per-step rate of **87%**; ten steps needs
**93%**. Nothing here is near that, dense included.

---

## 7. Interpreting the accuracy results

### 7.1 The ranking flips with budget, and both halves make mechanistic sense

**At 4096, TSA wins (p = 0.015 / 0.020).** This is the regime the method was designed for. When the
budget is tight, *which* chunks you pick dominates, and structure-aligned variable chunks avoid the
centroid-dilution failure. Consistent with this, `quest` vs `block` — same chunking, different
scoring — is **not significant** (p = 0.359). **Chunking is the variable that matters; scoring is
second-order once chunking is fixed.** That is a direct endorsement of TSA's core thesis.

**At 8192, TSA loses its advantage and is the only method still significantly behind dense
(p = 0.0042).** This is the uncomfortable result, and the budget-16384 run was added to resolve it.

### 7.1.1 The budget curves cross — and TSA reaches full-attention parity, but late

`idx_agree` as budget grows:

| Budget | TSA (tree) | Quest | BlockSparse |
|---|---|---|---|
| 4096 | **28.9%** | 20.5% | 18.4% |
| 8192 | 43.7% | **47.4%** | **47.9%** |
| 16384 | **51.1%** | not run | not run |
| *(full attention)* | *51.1%* | *51.1%* | *51.1%* |

Two things are happening, and they are mechanistically different.

**TSA degrades gracefully; fixed chunking has a threshold.** Going 4096 → 8192, TSA gains 14.8
points while Quest gains 26.9 and BlockSparse 29.5. Fixed 16-token chunking is *catastrophic* at a
tight budget and *excellent* once the budget clears some threshold. This is the signature of a
**resolution effect**: at 8192 the binding constraint stops being "did you find the right region"
and becomes "how precisely can you land on the right tokens", and page-16 has 4x the resolution of
page-64. TSA's 64-token pages buy coverage it no longer needs while paying resolution it now does.

**TSA is the only method demonstrated equivalent to full attention.** At 16384 it lands on 97/190 —
the identical count dense produced — with a ±3.3 point confidence interval and a 5/5 sign test. Quest
and BlockSparse at 8192 are *not shown different* from dense, but their intervals permit gaps up to
+9.5 points; that is a weaker statement, and it was never tested at 16384.

**The honest efficiency comparison is therefore unresolved at the top of the curve.** Directly:
`tree_k256` (16384) vs `quest_B8192` is 10/5, p = 0.30, CI [−0.017, +0.090] — **not significant**. So
TSA at twice the budget is not measurably better than Quest at half of it. To claim TSA is more
budget-efficient in this regime, Quest and BlockSparse must be run at 16384 too. **They have not
been, and until they are, no ranking above 8192 is supported.**

### 7.1.2 What survives

- **At tight budgets (4096), TSA is significantly better than both baselines.** Solid.
- **Chunking, not scoring, is the variable that matters** (quest vs block: p = 0.359 at 4096,
  p = 1.000 at 8192). Solid, and it is TSA's central claim.
- **Sparse attention can be lossless on this workload** — at 16384, i.e. ~55% of a 29.5k-token
  context, or ~2.7x compression. That is a genuine but modest compression ratio.
- **A caveat that has not been checked:** budget is `page_size × top_k` by construction, but TSA's
  chunks are *variable* (16–256 tokens) while its accounting is in units of 64. Whether TSA's
  realised token count matches its nominal budget as tightly as fixed chunking does is **unverified**.
  If TSA systematically attends to fewer real tokens per nominal budget, its 4096 win is understated
  and its 8192 loss is overstated. This is a one-hour instrumentation job and should be done before
  any of these numbers are published.

### 7.2 How much of the "failure" is the metric

Full attention scores 51.1% agree. Of its 190 index steps, **57 (30%) picked a valid but different
element**. Some of those are genuine mistakes; many are alternative valid paths.

So the practical ceiling for `agree` on this data is somewhere around 51–81% (agree ≤ ceiling ≤
valid), not 100%. **Sparse configs should be judged against ~51%, not against 100%** — which makes
`block_B8192`'s 47.9% look very different from how it reads in isolation.

### 7.3 What the accuracy results do *not* establish

Everything above is **offline replay agreement**, not task success. The reference context is
replayed at every step, so errors do not compound the way they would in a live agent loop. Earlier in
this project a live run showed exactly this gap: a replay probe estimated 46.4% invalid indices while
the live agentic loop produced 63.3%. **Online end-to-end success has not been measured for any of
these six configurations, and it is the metric that ultimately matters.**

---

## 8. Speed results

### 8.1 The one robust finding: context length does not change decode latency

Full attention, batch 1, ms per decode step, across three independently written harnesses:

| Harness | 25k ch | 50k | 75k | 101k | 118k |
|---|---|---|---|---|---|
| `speed_bench.py` | 138.6 | 155.6 | 152.8 | 139.5 | — |
| `bench2.py` | 150.5 | — | — | 140.9 | — |
| `bench4.py` | 155.5 | — | — | — | 153.5 |

Mean 148.1, sd 7.4, **no trend across a 4.7x KV range**, against a bandwidth prediction of +10.2 ms.
Two of the three harnesses used the estimator that §5.2 discredits, so this is *suggestive rather
than decisive* — but it replicates, and it is the observation the rest of §9 explains.

### 8.2 Where the 153.5 ms actually goes

Measured directly on the machine with real tensor shapes, batch 1, ~29.5k tokens:

| Component | ms/step | share | how obtained |
|---|---|---|---|
| **MoE experts (`grouped_mm`, the silent default)** | **77.1** | **50%** | measured, 1.607 ms/layer × 48 |
| q/k/v/o projections + norms + router + top-k | 13.1 | 9% | measured, 0.273 ms/layer × 48 |
| `lm_head` GEMV (2048 × 151936) | 3.4 | 2% | measured |
| **paged attention over the full 29.5k KV** | **≥13.0** | **≥8.5%** | roofline, 60.3 MB/layer ÷ 222.7 GB/s |
| sparse page selection (4 kernels + top-k, ×48) | 1–4 | 1–3% | inferred from launch count |
| residual: rope, residual adds, Python decode loop, per-token `argmax().item()` + `tokenizer.decode` | ~40–45 | ~28% | by subtraction |

**Attention is at most the fourth-largest term.** The 95% upper bound from the measurement is 22 ms
(15%); the roofline lower bound is 13 ms (8.5%). Both agree it is a minority cost.

### 8.3 The MoE kernel is the real bottleneck, and it is a one-line fix

`Qwen3VLMoeTextExperts` carries `@use_experts_implementation`, and with nothing specified,
`modeling_utils.py:1971` resolves to `"grouped_mm"`. Measured at batch 1 with the real shapes
(128 experts, gate_up [128,1536,2048], down [128,2048,768]):

| Backend | ms/layer | ms/step | vs default |
|---|---|---|---|
| `grouped_mm` | 1.607 | **77.1** | — (what actually ran) |
| **`batched_mm`** | **0.925** | **44.4** | **1.74x faster** |
| `eager` | 7.115 | 341.5 | 4.4x slower |
| roofline (8 active experts, 75.5 MB) | 0.339 | 16.3 | — |

Sweeping the total expert count with 8 active rows gave 1.887 / 1.672 / 1.580 / 1.634 / 1.209
ms/layer for E = 8/16/32/64/128 — **cost is independent of E**, which rules out "it materialises all
128 experts" (that would need 58 GB/step = 260 ms, more than the whole measured step). It reads ~8
experts' worth of bytes and takes 4.7x as long as that traffic costs: a CUTLASS grouped GEMM handed
128 groups of which 120 are empty, at M=1. An occupancy problem, not a bandwidth problem.

**Caveat:** at batch 8 the ordering reverses (`grouped_mm` 5.364 vs `batched_mm` 6.962 ms/layer). The
backend should be pinned *per batch size*, not globally flipped.

### 8.4 CUDA graph: capturable with `batched_mm`, but OOMs on real prompts

The §5.5 verdict was wrong. Establishing the real answer:

- `grouped_mm` **cannot** be graph-captured: `RuntimeError: Cannot copy between CPU and CUDA tensors
  during CUDA graph capture`. **This is why CUDA graph was disabled for the entire project** — and it
  is the same root cause as §8.3. One flag fixes both.
- `batched_mm` **can** be captured. The server logged `[CudaGraph] Capture complete`, `Captured for
  bs=1`, and answered a short request correctly.
- **But on a real 5,655-token browser prompt the container was OOM-killed** (`OOMKilled=true`, exit 0,
  zero Python errors). The graph memory pool plus 30B weights plus KV exceeded 119 GB unified memory.
- A retry with `--max-decode-tokens 512 --max-batch-size 1` is in flight to determine whether a
  reduced footprint makes it viable.

On a synthetic 8-layer stack, graph capture was worth **6%** (9.328 → 8.747 ms). The real model, with
rope, per-head norms, the attention kernel and 4 selector launches per layer, would likely see
10–15%. Either way it is second-order next to §8.3.

**Does CUDA graph change accuracy?** It should not — a graph replays identical kernels in identical
order, so outputs should be bit-identical. That was queued for empirical verification and is blocked
on the OOM above.

### 8.5 Per-config speed comparison — final

The v4 grid's per-config ranking was retracted (§5.3). The re-run used `bench5` (5 leveraged points ×
3 repeats, standard errors reported), batch 1 only, all ten configs on one machine. **Zero length
violations, all R² ≥ 0.99.**

**Batch > 1 is excluded** because `serve.py:178-190` performs 4 GPU→CPU synchronisations per request
per layer — 192·B forced stalls per step, 1,536 at B=8 — a materially different code path.

| Config | Budget | 25,036 ch | ±SE | 118,360 ch | ±SE |
|---|---|---|---|---|---|
| full (`topk=all`) | — | 134.9 | 7.6 | 134.5 | 5.6 |
| tree_B2048 | 2048 | 133.1 | 4.5 | 134.1 | 1.1 |
| quest_B2048 | 2048 | 132.2 | 2.5 | **126.7** | 6.6 |
| block_B2048 | 2048 | **129.1** | 6.2 | 135.1 | 4.9 |
| tree_B4096 | 4096 | 137.5 | 6.9 | 132.1 | 1.6 |
| quest_B4096 | 4096 | 140.4 | 7.0 | 136.5 | 1.3 |
| block_B4096 | 4096 | 134.6 | 2.5 | 131.3 | 1.0 |
| tree_B8192 | 8192 | **143.7** | 7.8 | **140.3** | 0.5 |
| quest_B8192 | 8192 | 142.6 | 6.8 | 136.6 | 4.1 |
| block_B8192 | 8192 | 129.8 | 3.3 | 134.9 | 5.6 |
| | | *spread 14.6* | | *spread 13.6* | |
| | | *sig. threshold ≈15* | | *≈9* | |

**Three conclusions, now on a validated estimator:**

1. **Context-invariance is settled.** Full attention: **134.9 ms at 25k chars, 134.5 ms at 118k** —
   a 4.7x increase in KV cache for a 0.4 ms change. There is no longer any interpretive room here.
2. **Sparsity buys nothing.** `tree_B2048` performs 12x sparsification and lands at 134.1 ms against
   full attention's 134.5. The spread across all ten configs (13.6–14.6 ms) is at or below the
   significance threshold implied by the measured standard errors (≈9–15 ms). **§9 predicted the
   spread would be under 15 ms before the data existed; it is 13.6 and 14.6.**
3. **The residual ordering is still anti-physical** — `tree_B8192`, which reads the *most* KV, is the
   slowest cell in both contexts — which is what noise looks like, not a mechanism. Do not report it
   as one.

### 8.6 The MoE-backend win did not survive to end-to-end (and the first test of it was void)

The §8.3 microbenchmark predicts `grouped_mm` → `batched_mm` saves 33 ms/step. A control was run —
and was **null by construction**: the env-var patch in `qwen3vl_inference.py` had never been applied
on that machine, so `TSA_EXPERTS_IMPL=batched_mm` was inert and *both* arms ran `grouped_mm`.

The accident is not worthless. Two nominally identical configurations differed by **−2.0 ms** at 25k
and **+4.9 ms** at 118k — an independent read on the noise floor that corroborates the ≈9–15 ms
threshold derived from the standard errors.

**The 33 ms claim is therefore untested end-to-end, not disproven.** A corrected control (patch
applied, backend line verified in the server log rather than assumed) is running. Until it returns,
§10-D1's "−21% from one environment variable" is a microbenchmark result only, and should be quoted
as such. Isolated microbenchmarks routinely fail to survive integration — overlap, memory state and
cache effects all differ — and end-to-end is the number that matters.

---

## 9. Why sparse attention cannot win here — the arithmetic

Amdahl's law: `speedup = T / (T − A + A·b)` where `A` is the attention time and `b = budget/context`.

| Situation | T (ms) | attention share | ceiling from *perfect* sparsity |
|---|---|---|---|
| As measured, 29.5k tokens | 153.5 | 8.5% | **1.086x** |
| As measured, 6k tokens | 155.5 | 1.8% | **1.012x** |
| + `batched_mm` | 95–120 | 11–14% | 1.12–1.16x |
| + CUDA graphs | 75–105 | 12–17% | 1.14–1.20x |
| + MoE GEMV at roofline | 45–60 | 22–29% | 1.28–1.40x |
| Ideal machine (100% bandwidth utilisation) | **40.3** | **32.3%** | **1.43x** |

**Even on a perfect machine the ceiling for this model at this context length is 1.43x.** On the
engine as it stands it is 1.09x — below the noise floor of any benchmark we could build in a night.

### 9.1 Comparison with Quest's published numbers

Quest (Tang, Zhao, Zhu, Xiao, Kasikci, Han — ICML 2024) reports **7.03x on the attention kernel
alone** (Fig. 12) and **1.74x end-to-end** in FP16 (Fig. 13; Llama-2-7B, 32k context, budget 2048).
*(The arXiv/PMLR abstracts transpose these two numbers; the paper body is correct.)* Our comparison is
end-to-end against end-to-end, so there is no category error — the gap is arithmetic:

| | Llama-2-7B @ 32k | Qwen3-VL-30B-A3B @ 29.5k |
|---|---|---|
| KV per token | 512 KiB (MHA, 32 KV heads) | 96 KiB (GQA 8:1, 4 KV heads) — **5.33x less** |
| Weights read per step | 13.48 GB | 6.083 GB (MoE, 3B active) — **2.22x less** |
| Total KV traffic | 16.8 GB | 2.90 GB |
| **Attention share of an ideal step** | **55.4%** | **32.3%** |
| Amdahl ceiling at budget 2048 | 2.16x (**1.74x measured**) | **1.43x** |
| Ceiling on *this engine* | — | **1.09x** |

Two multiplicative penalties:

- **0.66x from architecture** — GQA 8:1 shrinks the numerator, MoE shrinks the denominator less than
  it shrinks the numerator. **Not fixable.** It is what the model is.
- **0.76x from implementation** — the ~113 ms of context-independent overhead. **Fixable**, and §8.3
  is most of it.

**Nothing anomalous happened.** A method that delivers 1.74x on a dense MHA model at 32k delivering
~1.0x on a GQA-8:1 MoE at 29.5k is the predicted outcome, not a bug.

### 9.2 Where sparse attention *would* pay off on this model

- **Context length.** On an ideal machine, attention reaches 50% of the step at **~62,000 tokens**
  (~248k characters). On the current engine you would need ~356,000 tokens — outside the model's
  window. The 118k-character prompts in this dataset are ~29.5k tokens: **less than half of where
  this even becomes interesting.**
- **Batch size — the strongest lever, and this report previously had it backwards.** See §9.3.
- **Model choice.** A dense-MHA model would make the same method ~2.4x more impressive.

### 9.3 Batch size — a correction to this report

An earlier version of this document claimed "batch is not a lever; independent requests scale KV and
expert traffic together." **That is wrong.** The two traffics do not scale alike:

- **KV traffic is exactly linear in B.** Every request owns its cache; nothing is shared.
- **Weight traffic is sublinear and capped.** Each token activates 8 experts; different tokens in a
  batch may activate different ones, so expert traffic grows with B — but **saturates at 128
  experts**. Attention projections and `lm_head` are **constant** in B (shared across the batch).

Linear numerator over a sublinear, capped denominator → **attention's share rises with batch.**

At 29,500 tokens of context, budget 2048:

| B | distinct experts touched | weights GB | KV GB | **attention share** | Amdahl ceiling |
|---|---|---|---|---|---|
| 1 | 8 | 6.06 | 2.90 | **32.4%** | 1.43x |
| 2 | 15 | 9.23 | 5.80 | 38.6% | 1.56x |
| 4 | 28 | 15.12 | 11.60 | 43.4% | 1.68x |
| **8** | 50 | 25.08 | 23.20 | **48.0%** | **1.81x** |
| 16 | 80 | 38.67 | 46.40 | 54.5% | 2.03x |
| 32 | 110 | 52.26 | 92.80 | **64.0%** | 2.47x |
| 64 | 128 (saturated) | 60.42 | 185.60 | 75.4% | 3.36x |

**At B = 32, attention reaches 64% of the step — higher than Quest's 55.4% on Llama-2-7B** (§9.1).
The model is not inherently hostile to sparse attention; **the experiment was run at B = 1.** The
"1.43x ceiling" in §9 is a **batch-1 result**, and stating it without that qualifier was an error.

**Two caveats:**

1. **The distinct-expert column is an assumed overlap curve, not a measurement.** The true values
   depend on router behaviour on these browser prompts: if tokens across requests concentrate on the
   same experts, weight traffic grows more slowly and attention's share rises *faster*; if they
   spread out, the reverse. This is directly measurable (count distinct experts in the router's top-k
   across a batch) and cheap. Until then the table is a projection, not a prediction.
2. **Absolute B > 1 latencies are inflated by D3.** `serve.py:178-190` performs 4 GPU→CPU syncs per
   request per layer (192·B stalls per step). But those syncs live in the **selector**, which `full`
   also runs — so a **full-vs-sparse comparison at fixed B remains valid**, only the absolute numbers
   are inflated. §8.5's decision to exclude B > 1 entirely was **over-conservative**.

### 9.3.1 There *is* older batch data, and it settles nothing

The discarded v4 grid did sweep batch 1/2/4/8. It must not be used as evidence either way.

Full attention's context sensitivity — the direct test of whether attention is visible at all — by
batch, going 25k → 118k chars (4.7x more KV):

| B | 25k | 118k | change |
|---|---|---|---|
| 1 | 155.5 | 153.5 | **−1.3%** |
| 2 | 269.7 | 294.1 | **+9.0%** |
| 4 | 442.6 | 409.0 | **−7.6%** |
| 8 | 519.7 | 635.3 | **+22.2%** |

**Non-monotonic.** If this were the physical effect, B=4 would sit between B=2 and B=8; it is
negative instead. Other configs swing harder: `tree_B8192` shows **−28.5%** at B=2, `quest_B2048`
shows **+49.4%** at B=4 — 50-point swings between adjacent batch sizes of the same config.

The same applies to the sparse-vs-full question at B=8: `quest_B2048`, `block_B2048` and
`tree_B8192` all came out **7–8% faster** than full, while `tree_B2048` — which reads the *least* KV
and should therefore be fastest — came out **26% slower**.

This is §5.3's estimator defect showing up at B>1. Seizing on the +22.2% at B=8 while ignoring the
−7.6% at B=4 would be cherry-picking; so would citing "no speedup at large batch" from the same
table. **The v4 batch data can neither confirm nor refute §9.3.**

**Running now:** a batch sweep at 1/2/4/8 on the 118k context using the corrected bench5 estimator,
for `full`, `tree_B2048` and `quest_B2048`.

**Falsification criterion, stated before the data arrives:** full attention's 25k→118k increase must
rise **monotonically** with B. If B=8 still shows no context sensitivity under the good estimator,
the §9.3 arithmetic is wrong somewhere — most likely the expert-overlap assumption, since heavy
concentration on the same experts across a batch would make weight traffic grow far more slowly than
the table assumes.

---

## 10. Defects found, ranked

### Genuine bugs

**D1 — `experts_implementation` never pinned; the silent default is both the slow one and the
un-capturable one.** `models_qwen3vl_inference.py:137-140` only sets it from `TSA_EXPERTS_IMPL`, and
nothing set that variable. **Microbenchmark impact: −33 ms/step (−21%)** — 2.5x the entire attention
term at 29k and 12x it at 6k. It also blocks CUDA graphs (D2).
**Caveat (§8.6): this has not yet been confirmed end-to-end.** The first attempt to confirm it was
void, and isolated microbenchmarks frequently do not survive integration. Treat −21% as an upper
bound until the corrected control reports. Pin the backend per batch size, not globally.

**D2 — the CUDA-graph verdict was a harness artifact.** Single-reading death detection with no
debounce plus too-short post-restart sleep. See §5.5, §8.4.

**D3 — 4 forced GPU→CPU syncs per request per layer in the batched path.**
`serve.py:178,179,184,190` — including `int(pi_indptr[-1].item())`, a hard stall on a value the
selection kernel just wrote. 192·B per step; 1,536 at B=8. The batch-1 path has none. **All batch>1
measurements are of a different code path.** Fix: keep `page_indptr`/`last_page_len` on device.

**D4 — `page_bitset` silently truncates.** `csrc_ts_tree_sparse.cu:431` declares
`__shared__ uint32_t page_bitset[128]` (4,096 pages) and clamps with `min(..., 4095)`. At
`page_size=16` that caps at 65,536 tokens; beyond it, pages are **silently dropped** rather than
erroring. Not hit at 29k. `bitset_cap` is computed at `:429` and never used — the guard was
evidently intended and lost.

### Working as designed, but costly

**D5 — page selection runs on one SM with a single-threaded tail.** `csrc:896` launches
`select_pages_fused_kernel` with `grid(1), block(256)`; the collection loop at `:483-493` is
`if (tid == 0)` over `max_page`. At page 16 / 29k that is 1,842 serial iterations × 48 layers =
**88,416 dependent iterations per step**, vs 461 at page 64. Only ~0.5–1 ms — **but it is a
systematic penalty on exactly the Quest/BlockSparse arms, perfectly confounded with the
`--scoring-method` comparison those arms exist to make.** It biases *against* the baselines, i.e. in
TSA's favour, and should be fixed before publishing any speed comparison between them.

**D6 — the benchmark design could not resolve what it was built to measure.** §5.3.

**D7 — `torch.cuda.empty_cache()` in the per-request hot path.** `serve.py:493, 516, 651, 720, 794,
959`. A device sync plus `cudaFree` of every cached block, twice per request. It lands in the
intercept so it does not bias `ms_per_step`, but it is ~0.1–0.5 s of waste per request in production.

---

## 11. Conclusions

### On the research question

1. **TSA's core thesis is supported where it was designed to apply.** At a tight 4096 budget,
   structure-aligned variable chunking beats fixed chunking significantly (p = 0.015 / 0.020), while
   the scoring-function difference between the two baselines is not significant (p = 0.359).
   **Chunking is the variable that matters.**

2. **That advantage does not survive a looser budget, and TSA is the only method still significantly
   behind full attention at 8192.** This is the result that needs work, not a footnote. The two live
   hypotheses (granularity ceiling vs imperfect realised-budget matching) are separable with
   experiments that have not been run.

3. **Sparsity causes index hallucination.** Dense: 0/190. Sparse: 1–10%. Clean attribution.

4. **A third of the apparent failure is the metric.** Full attention itself only scores 51.1% agree,
   30% of its steps being valid-but-different choices. Judge sparse configs against ~51%, not 100%.

### On the speed question

5. **Sparse attention cannot pay off on this model at these context lengths, and no implementation
   fix changes that.** The ceiling from *perfect* sparsity is 1.09x now and 1.43x on a perfect
   machine. This is the predicted consequence of GQA 8:1 (5.33x less KV) plus MoE (2.22x smaller
   weight denominator), and it is consistent with Quest's own published 1.74x on a dense MHA model
   where attention is 55% of the step rather than 32%.

6. **The bottleneck is a MoE kernel selected by a silent default, not attention.** 77.1 ms of 153.5,
   4.7x off its own roofline, fixed by one environment variable worth ~33 ms. **The flag beats the
   algorithm.**

### Recommendations

**Immediate**
- Pin `experts_implementation` per batch size (D1). Free 21%, unblocks CUDA graphs.
- Fix D3's sync chain before trusting any batch>1 number.
- Fix D5 before publishing any TSA-vs-baseline *speed* comparison — it currently biases toward TSA.
- Add a real dense arm; rename `full` to `topk=all` (§5.6).

**For the research claim**
- **Stop claiming latency.** Position TSA on **accuracy per unit of budget**. That claim is supported
  at 4096 and is the honest one.
- **Resolve the 8192 regression.** First verify that TSA's realised token count matches its nominal
  budget as tightly as fixed chunking does — that is cheap and could explain the gap outright.
- **Measure online end-to-end success.** Every accuracy number here is offline replay agreement;
  the live agentic loop compounds errors and previously showed a 46% → 63% degradation.
- **If the goal is to demonstrate the method, change the setting:** a dense-MHA model, or contexts
  past 60k tokens. On Qwen3-VL-30B-A3B at 29.5k tokens the experiment is being run in the regime
  where the method structurally cannot win.

---

## Appendix — reproduction

| Artifact | Path |
|---|---|
| Per-step accuracy results | `simulator/runs/sparse3way-20260721/result_h1_*.jsonl` |
| Full-attention baseline | `simulator/runs/sparse3way-20260721/result_h1_dense.jsonl` |
| Speed grid (v4, superseded) | `simulator/runs/sparse3way-20260721/speedbench/bench4_all.jsonl` |
| Evaluation datasets | `offline_half1.jsonl` (50 tasks / 332 steps), `half1_task_ids.json` |
| Benchmark harness | `bench5.py` (`/workspace/speedbench/` on spark01) |
| Server | `/workspace/TreeSparseAttention/serve.py` on spark01 |

Server launch pattern (all configs differ only in the flags shown in §2.3):

```bash
TSA_EXPERTS_IMPL=batched_mm python3 serve.py \
  --model-path /models/Qwen3-VL-30B-A3B-Instruct --host 0.0.0.0 --port 10000 \
  --max-decode-tokens 4096 --max-batch-size 8 --batch-collect-ms 150 \
  --served-model-name tree-sparse \
  --page-size 64 --top-k 64 --tree-parse-mode webarena --scoring-method envelope
```

**Still running at the time of writing:** the `bench5` speed grid with `batched_mm` (§8.5), the
reduced-footprint CUDA-graph retry (§8.4), and `tree_k256` at budget 16384 (§7.1). This document will
be updated when they land.
