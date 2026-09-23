# Sparse Attention for Browser Agents: the BrowserSparseAttention Project Report

**Date**: 2026-09-03 (experiment window 2026-06 → 2026-09-02)
**Model**: Qwen3-VL-30B-A3B-Instruct (48 layers, 32 query heads / 4 KV heads, GQA group size 8, head_dim 128, MoE with 30B total / 3B active parameters), text-only
**Hardware**: NVIDIA GB10 (sm_121, 119 GB unified memory), spark00 / spark01
**Code**: `BrowserSparseAttention` (formerly TreeSparseAttention; `main` @ `60c4013`, branches `shiqihe/region-aware` @ `5afd993` and `shiqihe/mixmax-main` @ `3bbb01b`)
**Companion documents** (this report is a standalone synthesis of them; none needs to be read first): `2026-08-23_scoring_function_study.zh.md` (full details of the scoring functions and of the two implementation bugs), `2026-08-28_mixmax_head_aggregation_design.zh.md` (attribution of contributions and related work), `2026-09-01_codesign_report_mixmax_region_aware.zh.md` (the region-aware and co-design experiments), `2026-08-07_speed_study_tsa_vs_quest_vs_block.en.md` / `2026-08-07_speed_study_tsa_vs_quest_vs_block.zh.md` (the early-August speed study), `2026-09-02_chunking_comparison.zh.md` (five chunking schemes compared on the same prompt)
**Figures**: `figures/bsa_report/fig01`–`fig17` (reproducible with `make_figures.py`; all numbers come from the documents above and from the raw results under `runs/sparse3way-20260721/`)

---

## How to read this report

This report is written for **researchers who have not worked with sparse attention before**. §1 uses a 12-token toy example to explain what sparse attention is, at which point of generation it happens, and how the conventional methods (Quest, BlockSparse, etc.) do it; §2 explains why the prompt structure of a browser agent defeats the conventional approach; §3 introduces the four components of this project one by one — (1) variable-length semantic chunking, (2) a new scoring function, (3) averaging across query heads, (4) mixed granularity for system prompt / history versus DOM together with budget allocation — each with a toy example, code locations, and ablation evidence; §4–§5 are the experimental setup and the results; §6 summarizes the differences from the conventional methods in one table; §7 gives the boundaries of the contributions and the follow-up directions. Readers who only want the conclusions can read §0 and §6.

## Glossary (read this first)

| Term | Definition |
|---|---|
| **prefill / decode** | An LLM processes a request in two phases: prefill reads the whole prompt in one pass; decode generates the output tokens one at a time. Generating one token is one **decode step**. |
| **KV cache** | During prefill, every token in the prompt produces one key/value vector pair at every layer and every KV head, which is cached; during decode, the query of the new token is dotted with these keys to decide "where to look". For this model that is about 98 KB per token (48 layers × 4 KV heads × 128 dims × 2 vectors × bf16), so a 20k-token page is about 2 GB. |
| **full attention (dense)** | Every decode step, every layer reads the entire KV cache. |
| **sparse attention** | Every decode step, every layer reads only a subset of the KV cache. This report only discusses **query-aware selection** (the subset is chosen dynamically from the current query and can differ at every step), not eviction methods that permanently delete KV entries (H2O, SnapKV). |
| **attention head / GQA** | Each layer has 32 query heads; grouped-query attention (GQA) makes every 8 query heads share 1 KV head, giving 4 groups. When selecting KV, one must decide whether these 32 heads each select their own set or share one selected set. |
| **chunk / page / block** | **chunk** = the unit of scoring and selection in this report (a contiguous span of tokens). **page** = the storage unit of KV in GPU memory (FlashInfer paged KV; page_size = 16 or 64 in this project); the attention kernel reads by page, and a selected chunk is mapped to the pages it covers. Quest's page and Vortex's block are both "fixed 16-token chunks". |
| **budget (B)** | The upper limit on the number of KV tokens that may be read per decode step, per layer, e.g. B = 2048. A budget can be counted in chunks (top-k) or in tokens; the two differ a lot under variable-length chunks (§3.4). |
| **always-include** | The part that is read regardless of scores: the attention sink (the first 4 tokens of the sequence), the most recent 128 tokens, and all tokens generated so far in this request. Identical for all methods. |
| **summary** | A cheap statistic computed at prefill for every chunk, every layer, every KV head; at decode it alone is used for scoring, without reading the keys. Two kinds: **envelope** = the per-dimension max vector M and the per-dimension min vector m; **centroid** = the mean of the keys. |
| **upper bound scoring** | Using the envelope to compute Σ_d max(q_d·M_d, q_d·m_d): the dot product of q with **any** key inside the chunk cannot exceed it, hence "upper bound". It answers "how relevant, at most, is the best key in this chunk". |
| **width term** | The second term of the equivalent rewriting q·(M+m)/2 + Σ_d \|q_d\|·(M−m)_d/2; (M−m) is the spread of the keys in the chunk along each dimension, which naturally grows with chunk length. The first term is called the center term. |
| **centroid scoring** | q·mean(k): answers "how relevant is this chunk on average". |
| **dilution** | The failure mode of the centroid: when only one line in the chunk is the target, the mean averages it away. |
| **width bias** | The failure mode of the upper bound: a large, heterogeneous chunk scores high because its width term is large, even when it contains no matching key at all. |
| **needle / needle hit rate** | The chunk containing the element that the reference trajectory actually clicked at that step (e.g. `[58]<input …>`) is called the needle; the **needle hit rate** = the fraction of selections (per layer, per decode step) that put it inside the budget. If the model cannot read that line it cannot output the correct index, so this is a necessary condition for correct behaviour. |
| **attention mass / mass recall** | The softmax weights of full attention; the mass of a chunk = the sum of the weights of its tokens; **mass recall** = the fraction of the total weight covered by the selected chunks. Mass concentrates heavily on the sink and on the instructions, and the needle usually holds only a tiny share — the two objectives often conflict. |
| **agree / valid / none** | Offline replay metrics. The original context of a step of the reference trajectory is fed unchanged to the configuration under test (**teacher-forced**); if the output element index matches the reference, it counts as **agree**; if the index exists on the page, it counts as **valid**; if no index action is output, it counts as **none**. The denominator contains only index-action steps. agree underestimates the true correctness rate (choosing a different but equally reasonable element also counts as a disagreement), and it is not the same as online task success rate. |
| **fidelity** | The fraction of steps on which the configuration under test outputs the same action as full attention. |
| **run-to-run variance** | Re-running full attention itself changes about 20% of the outputs (batch composition, numerical non-determinism); with n≈130–190, differences of fewer than 5 steps are not interpretable, and all comparisons use paired sign tests and bootstrap intervals. |
| **transparent regime** | The range of budgets large enough that differences between scoring functions are invisible in end-to-end metrics (B ≥ 3k–4k tokens for this workload). |
| **region** | A semantic region of the prompt: system prompt, task and history, DOM observation, trailer. |
| **budget floor** | A fixed fraction of the budget reserved for one region, within which chunks are still chosen by score; notation sf25 = system-prompt floor 0.25. |
| **admission** | The packing step after scoring: candidate chunks are packed into the budget in descending order of score, skipped if they do not fit, until the budget is exhausted (greedy knapsack). |

---

## 0. One-page summary

**What sparse attention is and when it happens**: For every token an LLM generates, the attention in every layer has to read the entire KV cache once. Sparse attention stores a summary for every chunk of the KV cache at prefill, then at **every decode step, in every layer**, scores all chunks with the current query and reads only the highest-scoring ones, up to a total not exceeding the budget. It deletes no KV; the next step may select a completely different subset. In this project's implementation, this selection happens inside the forward function of every layer's attention (`models/tree_sparse_patch.py:531-543`), and the summaries are computed once per layer at prefill (`python/tree_sparse_selector.py:562-641`).

**The conventional approach** (Quest, Vortex BlockSparse, LServe, ArkVale, etc.): cut the KV into fixed-size pages of 16 tokens each, regardless of content; store an envelope or a centroid per page; every head scores on its own and selects its own top-k pages; count the budget in pages.

**What is different about a browser agent's prompt**: it has a fixed structure — a system prompt of about 5.3k tokens (rules, output format), a task-and-history section of a few hundred to over ten thousand tokens, and a DOM observation of 0.7k–6k tokens (one page element per line). The action target (the element to click) **can only be in the DOM region**, and its minimal semantic unit is "one element line"; the system prompt only needs to be covered, not resolved at fine granularity. A fixed 16-token page splits one element line into 2–4 pages (Figure 3), with the index and the text landing in different pages; the tidy page numbers also hide the variable that actually decides success or failure — "how much budget does each region get".

**This project does four things** (§3):

1. **Variable-length semantic chunking**: chunk along the ChatML / DOM tree, so that one DOM element subtree is one chunk; the region-aware version uses element-level chunks of 6–64 tokens in the DOM region and coarse chunks of 64–256 tokens in the instruction/history regions. On the same 11k-token prompt, the number of selection units drops from 692 under fixed-16 to 204 (Figure 14).
2. **The scoring function mixmax_wn**: fit Quest's per-dimension upper bound into the existing kernel structure, and multiply the width term by (16/L)^0.25 to cancel the systematic over-estimation of long chunks. This is the key that makes variable-length chunks usable: with the same chunking and no normalization, end-to-end is significantly worse (89 vs 96, p = 0.02); with normalization it is statistically indistinguishable from fixed-16 and from full attention (Figure 9). The normalization exponent 0.25 agrees with the width growth exponent β = 0.245 measured afterwards directly on real KV (Figure 6).
3. **Averaging across query heads**: the 8 query heads within a GQA group are first averaged into one vector, then the scores of the 4 KV heads are averaged, so that all 32 heads of a layer share one selected set. The cost is that the retrieval signal living in a few heads gets diluted; measured end-to-end, there is no difference from head-max at B ≥ 3k and a measurable cost at B ≤ 2k; in exchange, scoring compute drops 8× and a single page table compatible with serving engines suffices.
4. **Mixed granularity + budget allocation**: greedy admission counted in tokens, plus a system-prompt floor of 0.25 that guarantees instruction coverage. At B2048 this raises method 1 from 48 to 57 (= full attention; the fixed-16 baseline is 55), and on half1 from 82 → 94 (Quest 91, dense 97); at B1024 it ties the baseline in the paired comparison (53 vs 55), but its format compliance is significantly better (none 8 vs 31).

**Efficiency**: on the CUDA selector, the per-layer, per-step selection time is 51.7–65.0 µs (region-aware) versus 74.1–114.0 µs for fixed-16, i.e. 1.4–1.75× faster (Figure 15). But on this machine and this MoE model, attention accounts for ≤ 15% of a decode step, so the end-to-end speedup ceiling of any sparse attention is only about 1.09× (Figure 16); the benefit of sparse attention only becomes significant on dense MHA models, longer contexts, or larger batches (upstream measured 1.74–3.53× on B200 / Qwen3-VL-8B).

**What is not claimed**: the upper bound formula (Quest), head pooling (pre-existing in TSA), semantic tree chunking itself (pre-existing in TSA), and cross-step reuse of selections (LServe) are not contributions of this project; nor does this project **beat** the fixed-16 baseline in accuracy (all differences are within run-to-run variance). The claim is "ties it at the same amount of KV read, with 3–5× fewer scoring units and with structural information made available". The directions that have been refuted (hierarchical descent, DOM head-max, priority admission of interactive elements, hard three-region partition) are in §5.5.

---

## 1. What sparse attention is

### 1.1 What happens when one token is generated

![](figures/bsa_report/fig01_decode_step.png)

*Figure 1: What full attention (a) and query-aware sparse attention (b) each read within one decode step. In (b), the summary M, m of every chunk is computed at prefill; at decode, only q and the summaries are used for scoring, and only the selected chunks (orange boxes) plus the always-include sink and recent tokens (blue) are actually read.*

In every layer, attention does three things for the current token: it computes a query vector q from the token's hidden state; it takes the dot product q·k_i with the key k_i of every earlier token in the KV cache and turns the results into weights with a softmax; it sums the corresponding values v_i weighted by those weights. The k and v of the prompt are computed once at prefill and cached, and never change afterwards; q is new at every decode step, in every layer and every head — it encodes "what is being looked for now". When the model is about to write `"index": 58`, the q of certain heads will have a very high dot product with the k of the few tokens of `[58]<input …>` in the DOM.

**Where the cost is**: for every generated token, every layer has to read all n (k, v) pairs from GPU memory once. With a 20k-token context that is about 2 GB per step; this is the quantity sparse attention is meant to cut.

**What sparse attention does** (Figure 1b):

- **At prefill**, compute a summary for every chunk, every layer, every KV head: an envelope (per-dimension max/min) or a centroid. In this project: `python/tree_sparse_selector.py:562-641` `compute_centroids`, called once per layer; the summaries are quantized to fp8 for storage (the whole summary is read once at every decode step in every layer, and fp8 halves that traffic).
- **At every decode step, in every layer**: use the layer's q to score all chunks, sort them, pack them into the budget, then map the selected chunks to pages and run attention on those pages only. Yes, **the selection is redone at every decode step and in every layer** — because q differs at every step and every layer. Code locations: `models/tree_sparse_patch.py:531-543` (`selector.select_pages(query=q, layer_id=…)` is called inside every layer's attention forward), `models/direct_decode.py:1194-1198` (the fast path without HF dispatch). In CUDA-graph mode there is a variant: so that the whole forward can be captured as a static graph, the selection is precomputed outside the graph using the q of the **previous** step (`direct_decode.py:813-830` `_pre_pass`, "lagged query").
- **always-include**: the sink (first 4 tokens), the most recent 128 tokens, and the generated tokens are always read (`TreeSparseConfig` defaults, selector `:65-66`; packing logic `:1349-1380`).
- **Budget accounting**: by chunk count (top-k, original TSA) or by token count (added in this project, `:328-375`).

The quality of the selection depends on only two things: **how the chunks are cut** (chunk geometry) and **how the summaries are scored**. A wrong selection = the chunk holding the keys with truly high dot products is ranked outside the budget, and the model cannot see that element.

### 1.2 How conventional sparse attention does it

| Method | Selection unit | Summary | Scoring | Head handling | Budget |
|---|---|---|---|---|---|
| **Quest** (Tang et al., ICML 2024) | fixed 16-token page | per-dimension min/max envelope | Σ_d max(q_d M_d, q_d m_d) (upper bound) | every head scores on its own and selects its own top-K pages | page count |
| **BlockSparse** (Vortex, Infini-AI-Lab) | fixed page | centroid | q·centroid → softmax (scale 0.09) | per-head softmax, max within the group, pages selected per KV head | page count |
| **LServe** (MLSys 2025) | fixed hierarchical pages (physical page = g logical pages) | logical-page min/max, max over the physical page | same as Quest | — | page count; the selection result is reused across ≤ 8 steps |
| **ArkVale** (NeurIPS 2024) | fixed page | bounding-volume digest (a generalization of the envelope) | Quest-like | per head | page count |
| **ShadowKV** (ICML 2025) | fixed chunk | mean-pooled key (centroid-like) | dot product | low-rank / pooled | block count |
| **This project** | **variable-length semantic chunk (aligned to the DOM/ChatML tree, granularity set per region)** | envelope (fp8) | Quest bound + width normalization | mean within the GQA group, mean across KV heads, shared selected set | **token count + region floor** |

What they have in common: all use fixed-size, content-agnostic units; all reselect at every decode step, in every layer; all keep the sink and the recent window. LServe itself points out that "Quest fails when page sizes increase" — once a fixed page gets larger, the envelope becomes loose and accuracy drops; this is exactly the problem that variable-length chunks must face (§3.2).

### 1.3 Toy example: how three scoring functions differ on the same numbers

Let head_dim D = 4, with 12 tokens cut into 3 chunks of 4 tokens each, and the current q = [2, −2, 1, −1].

| chunk | 4 keys | q·k (the logits of full attention) | Real-world counterpart |
|---|---|---|---|
| **A** (needle) | [2,−2,1,−1] · [0,0,0,0] · [−1,1,0,0] · [0,0,−1,1] | **10**, 0, −4, −2 | one strongly matching key + three irrelevant tokens: the DOM subtree containing the target element |
| **B** (large and heterogeneous) | [3,3,0,0] · [−3,−3,0,0] · [0,0,3,3] · [0,0,−3,−3] | 0, 0, 0, 0 | no key matches q at all: a navigation bar, a pile of links |
| **C** (small and homogeneous) | [1,−1,1,−1] × 4 | 6, 6, 6, 6 | four nearly identical, moderately matching keys: a cookie button |

Full attention takes a softmax over the 12 logits: the first key of A receives **93.2%** of the weight, C gets 6.8% in total, B gets 0.02% in total. When the budget only allows reading 1 chunk, the correct answer is A.

The summaries computed at prefill (independent of q):

| chunk | M (per-dim max) | m (per-dim min) | c = (M+m)/2 | w = (M−m)/2 | centroid |
|---|---|---|---|---|---|
| A | [2,1,1,1] | [−1,−2,−1,−1] | [.5,−.5,0,0] | [1.5,1.5,1,1] | [.25,−.25,0,0] |
| B | [3,3,3,3] | [−3,−3,−3,−3] | [0,0,0,0] | [3,3,3,3] | [0,0,0,0] |
| C | [1,−1,1,−1] | [1,−1,1,−1] | [1,−1,1,−1] | [0,0,0,0] | [1,−1,1,−1] |

![](figures/bsa_report/fig04_toy_scoring.png)

*Figure 4: The scores that five scoring functions give A / B / C on the same numbers. Black is the true maximum dot product within the chunk.*

| Scoring | Formula | A | B | C | Selected when budget = 1 |
|---|---|---|---|---|---|
| Quest upper bound | Σ_d max(q_d M_d, q_d m_d) = q·c + \|q\|·w | 2 + 8 = **10** | 0 + 18 = **18** | 6 | B whenever B is present ✗ |
| Original TSA 2-corner | max(q·M, q·m) = q·c + \|q·w\| | max(2, 2) = **2** | 0 | 6 | C ✗ |
| centroid | q·mean | **1** | 0 | 6 | C ✗ |
| mixmax_wn (A and C are 16 tokens, B is 256 tokens) | q·c + (16/L)^0.25·\|q\|·w | 10 | 0 + 18 × 0.5 = 9 | 6 | **A ✓** |

Five observations, each corresponding to a mechanism discussed later:

1. **Quest's 10 for A is exactly the dot product of the true strongest key**. The per-dimension max is the exact upper bound of q·k over the chunk's bounding box [m_d, M_d]; A's needle is the extreme in every dimension, so the bound is tight. This is the value of the envelope: a single strongly matching key is enough to lift the whole chunk.
2. **2-corner gives A only 2**. When the signs of q are mixed across dimensions (+, −, +, −), summing per dimension first and then taking the max lets the width term cancel out (3 − 3 + 1 − 1 = 0), leaving only the center term — mathematically it degenerates into "scoring with the chunk midpoint" and loses its sensitivity to a single strong key (§3.2).
3. **The centroid gives A a 1**: the needle is averaged away by three irrelevant tokens, i.e. dilution.
4. **Quest gives B 18, higher than A**: B contains no matching key at all, but its bounding box is large (w = 3), and \|q\|·w hands it 18 points for free — i.e. width bias. The upper bound is tight only when the chunk really has a key sitting at the optimistic corner.
5. **C scores 6 under every scoring function**: a small homogeneous chunk has a bounding-box width of 0, and its centroid is the key itself — the three scoring functions do not differ. Most fixed-16 pages are close to type C, which is why scoring functions differ very little on fixed-16 (the transparent regime of §5.1); variable-length chunks create type A and type B at the same time, and both failure modes are amplified together.

The width normalization in row 6 (§3.2) is this project's fix for point 4: if B is a large 256-token chunk and A is a small 16-token chunk, multiplying B's width term by (16/256)^0.25 = 0.5 restores the ordering.

---

## 2. What is different about a browser agent's prompt

### 2.1 Prompt structure and where the attention goes

![](figures/bsa_report/fig02_prompt_anatomy.png)

*Figure 2: (a) The structure of the prompt that the browser-use agent sends to the LLM at each step (two typical sizes); (b) the distribution of full-attention weight by region (average over 15 real steps and all layers, `region_sweep.py`).*

At every step, the agent hands the LLM a prompt of the following form and asks for one JSON action (containing the target element index):

```
[system prompt]    ~5.3k tokens: operating rules, output format, example DOM   —— can never be the action target
[task + history]   a few hundred to 13k tokens: the goal and all previous steps  —— likewise never the action target
[DOM observation]  0.7k–6k tokens: the list of page elements, one element per line
                   e.g. [58]<input placeholder=Find a recipe …>                  —— the action target (needle) is only here
[URL etc. trailer] a few dozen tokens
```

Three facts determine the design:

- **The action target is only in the DOM region, and its minimal semantic unit is "one element line"**: the index `[58]` and its text must be read together for the model to output 58.
- **The system prompt only needs coverage (mass), not granularity**: it receives 22% of the attention mass but is never an action target; cutting it finely is pointless, while dropping it makes the model lose the output format (§5.3, mechanism B).
- **Needle and mass are two conflicting objectives**: mass is dominated by the sink (54%) and by the instructions, the DOM as a whole gets only 12%, and the needle is just one line inside the DOM. Methods that select chunks by mass (the centroid family, original TSA) spend the budget on the instructions and miss the needle; methods that select by upper bound do the opposite.

### 2.2 The problem with fixed pages on the DOM: a real example

![](figures/bsa_report/fig03_real_chunking.png)

*Figure 3: The boundaries under five chunking schemes for tokens 9584–9866 of the DOM region of the same real prompt (the Allrecipes home page, step 2 of a WebVoyager trajectory, 11,062 tokens). The red area is this step's needle — the search box `[58]<input>`. Data from `CHUNKED_*.md`.*

- **fixed-16**: the line `[58]<input …>` (about 37 tokens) spans 4 pages, with the index `[58]` in one page and the placeholder text in two others. To "read this element", 3–4 pages (48–64 tokens) must be selected at the same time; each page's summary describes only half an element.
- **The old semantic leaf chunking (one parameter set, 16/256)**: the element is merged into a large 91-token chunk (in the same chunk as the Newsletters link, etc.), and there are also 2-token fragments (the whitespace tokens of the parent node between leaves) — the large chunk dilutes the needle and the fragments waste units, the worst of both (the chunking root of mechanism A in §5.3).
- **Subtree chunking on main (16/32)**: the element is split into two halves, [9759–9790] and [9791–9822] — an upper limit of 32 is still too small for a DOM element line.
- **region-aware (this project)**: `[58]<input>` + `[113]<button Search>` + `[814]<div>` form one 64-token chunk — the search box and the search button happen to be the two candidate actions of this step, and they are selected together.
- **V2 prototype** (content-driven chunking proposed upstream, not deployed): aligned to lines and brackets, element-level, but with the largest number of units (446).

### 2.3 Operating point

Context of 6k–23k tokens; a fresh prefill at every step (the history is re-sent as text); short generation (one JSON action, a few dozen to a thousand tokens). The budget range of interest is 1k–4k tokens (about 1/4 to 1/20 of the context).

---

## 3. Method: the four components of Browser Sparse Attention

### 3.1 Component 1: variable-length semantic chunking

**Tree**: `python/tree_parser.py:90` `parse_webarena_tree` parses the prompt's token stream into a tree — the ChatML turns (system / user / assistant) form the top level, every line `[N]<tag …>` in the DOM observation becomes an `actree_node` according to its indentation depth, and the remaining text becomes nodes by paragraph.

**From tree to chunks** (`tree_parser.py:880-981` `extract_subtree_chunks`): a whole subtree (tag + content + closing) that does not exceed max is packed into one chunk; nodes that are too large are descended into, and their leading/trailing residual tokens are attached back to the first/last child chunk (tags do not cross nodes); small chunks are merged only within the same parent node (never across siblings); the returned chunks are guaranteed to tile the whole sequence exactly.

**region-aware** (`tree_parser.py:1015-1069` `extract_region_chunks`):

1. Use a regular expression to find the first/last `[N]<` after the first `<|im_start|>user`, which defines the DOM span [dom_lo, dom_hi] (`:998-1012` `find_dom_token_span`);
2. Run subtree chunking over the whole text once at the fine setting (6, 64);
3. Adjacent chunks that do not overlap the DOM span are greedily merged up to ≥ 64 and then hard-cut at ≤ 256; chunks inside the DOM span are kept as they are.

The result is two granularities — "coarse chunks in the instruction/history regions, element chunks in the DOM region" — enabled by `TSA_HYBRID_GEOM=1` (standalone) or `BSA2_CHUNK_EXTRACT=region` (sglang), with parameters `TSA_DOM_MIN/MAX = 6/64` and `TSA_OTHER_MIN/MAX = 64/256` (selector `:198-207`).

![](figures/bsa_report/fig14_chunk_counts.png)

*Figure 14: (a) The number of units for the same 11,062-token prompt under six chunking schemes (`2026-09-02_chunking_comparison.zh.md`); (b) the number of scoring units for the two real prompts in the selection micro-benchmark.*

Evolution of the chunking: the old leaf chunking (leaves and the gaps between leaves as separate chunks, gaps never merged) produced 103 fragments of < 16 tokens on one prompt and has been replaced by subtree chunking; the subtree chunking on main with max = 32 makes 82% of the boundaries in the prose regions degenerate to a fixed stride; region-aware uses an upper limit of 64 in the DOM region and 256 in the instruction region, giving 204 units, and is the only one of the three that satisfies both "elements are not split in the middle" and "coarse chunks in the instruction region".

**Why not simply enlarge Quest's pages**: uniformly large pages pay a "granularity tax" everywhere — selecting a 64-token page to get a 6-token element wastes 58 tokens of budget, and the page boundaries still have nothing to do with elements; variable-length chunks are large only where the content is semantically coherent, and their boundaries align with elements. A direct comparison against fixed-64/32 at the same budget has not been measured on this workload and is an ablation that should be added.

### 3.2 Component 2: the scoring function mixmax_wn

Three formulas (for each KV head h, the group-mean query q̄_h; the chunk's envelope M, m and length L):

```
2-corner (original TSA):  s = max( q̄·M , q̄·m )
mixmax:                   s = q̄·(M+m)/2 + Σ_d |q̄_d|·(M−m)_d/2          ← identical to Quest's per-dimension upper bound
mixmax_wn:                s = q̄·(M+m)/2 + (16/L)^0.25 · Σ_d |q̄_d|·(M−m)_d/2
```

**Why mixmax is Quest's formula**: the scalar identity max(a, b) = (a+b)/2 + \|a−b\|/2, with a = q_d M_d and b = q_d m_d and using M_d ≥ m_d, gives max(q_d M_d, q_d m_d) = q_d(M_d+m_d)/2 + \|q_d\|(M_d−m_d)/2; summing over d yields mixmax. Numerical check (D = 2, q = (2, −1), M = (3, 5), m = (1, −4)): Quest = max(6, 2) + max(−5, 4) = 10; mixmax = 3.5 + 6.5 = 10; whereas 2-corner = max(q·M, q·m) = max(1, 6) = 6 < 10 — taking the max over whole vectors forces every dimension to pick the same corner, and the bound becomes weaker; this is the degeneration of point 2 in §1.3. Two engineering reasons for rewriting it as center + width: the width term is exposed separately, so the normalization has something to act on; and c, w are precomputed once per chunk, so scoring needs only two GEMMs (the `ctr`/`wid` lines of `_score_mixmax`, selector `:1338-1339`).

| Component | Source | Contribution of this project? |
|---|---|---|
| per-dimension upper bound | Quest, unchanged | No |
| GQA head-mean (mean within the group + mean across KV heads) | pre-existing in TSA, common in the field | No (only ablation evidence is provided) |
| **length normalization of the width term, (16/L)^α** | this project | **Yes**; in the literature, all bound-based methods use fixed-length units and all variable-length-unit methods use centroids (ClusterKV, Tactic; DHSA applies a √L scaling to a mean aggregate, a different object) — this cell is empty |

**Why normalization is needed — the origin of width bias**: the width term is determined by the per-dimension extremes of the keys in the chunk; the extreme of L samples grows with L (iid Gaussian approximation ~√(2 ln L)), so the upper bound of a long chunk is systematically inflated, regardless of whether it contains the target; the center term does not inflate systematically with L. The correction should therefore act only on the width, decrease monotonically with L, and vary slowly.

![](figures/bsa_report/fig06_width_growth.png)

*Figure 6: The growth of the width term with L, measured directly on real web-agent KV dumps (2 sets of dumps × 3 requests × 4 layers, 150 random contiguous spans per length bin; `scoring_case_study/measure_width.py`). The fitted slope is β = 0.245 (range 0.192–0.279 per dump × layer), coinciding with the deployed α = 0.25; after correction, the curve is flat (±7%) over the whole range L = 8–256. The measured inflation at 256 tokens is 1.86×, faster than the 1.41× of the iid theory — the excess comes from the content heterogeneity of semantic chunks.*

**Choice of α**: first swept over {0, 0.25, 0.5, 0.75, 1} by task metric on failure-enriched dumps (Figure 7), later verified independently by the geometric measurement of Figure 6. The anchor 16 = the scale of the baseline page: a chunk with L = 16 gets a discount of 1 (the score scale is untouched, and when degenerated to fixed-16 it equals Quest bit for bit), L = 64 → 0.707, L = 256 → 0.5, and a 6-token element chunk → 1.28.

![](figures/bsa_report/fig07_alpha_sweep.png)

*Figure 7: The α sweep (tree chunks, 15 real steps, `sweep_wn.md`). (a) When the budget is counted in chunks, α = 0 is best — the width term itself is a needle signal; when the budget is counted in tokens, α = 0 collapses (0.18) and α = 0.25 is best (0.81). (b) The reason: the chunks selected at α = 0 average 108 tokens (corpus mean length 39), so the same top-64 actually reads 1.6× more KV — width bias amounts to a hidden budget inflation.*

![](figures/bsa_report/fig08_offline_budget_curves.png)

*Figure 8: Offline needle hit rate as a function of token budget (`sweep_v3.md` / `sweep_wn.md`). The un-normalized Quest bound almost fails on variable-length chunks at B ≤ 4k (0.04–0.13); after normalization (green) it is no lower than either of the two fixed-16 baselines at every budget.*

**End-to-end ablation** (half1, 190 steps, B4096, paired): with the same chunking, tree + Quest bound without normalization = 89 (vs fixed-16 Block 99, sign test p = 0.02, significantly worse); mixmax_wn = 96 (p = 0.65, indistinguishable). That is, the normalization moves variable-length chunks from significantly worse into the run-to-run variance range of full attention.

**Code**: PyTorch reference `python/tree_sparse_selector.py:1327-1347` `_score_mixmax`; CUDA kernel `csrc/ts_tree_sparse.cu:402` `score_fp8_mixmax_kernel` (Jaccard 0.99–1.00 with the reference implementation on the selected page sets); sglang side `bsa_sglang/selection.py:66-93` `score_chunks_mixmax`. The summaries are stored in fp8 (e4m3 + per-vector scale), with no visible effect on the ranking in measurements (`legacy_fp32 ≈ legacy`, Δ ≤ 0.002).

### 3.3 Component 3: averaging across query heads

**Data flow** (the formulas write only one q̄, but there are actually two averaging steps):

```
32 query heads
  → (first averaging) the 8 heads within each GQA group are averaged into 1 → 4 q̄_h        (selector :1153 _group_query_heads)
  → each q̄_h computes the bound with the envelope of its own KV head → 4 scores per chunk
  → (second averaging) the mean of the 4 scores → 1 score per chunk                          (selector :1347  (ctr + wid).mean(dim=1))
  → admission → all 32 heads of the layer share the same selected set
```

**Why this is done**: the serving kernel gathers only **one** shared page set per layer — the block tables of mainstream serving engines (sglang / vLLM) are per-sequence, and there is no decode kernel with per-head page tables; the selection path (top-k, index construction, reselect cache) is done once instead of 4 times; 4 dot products per unit instead of 32. Original Quest selects its own top-k pages per head and relies on its own research-prototype kernel; the fixed-16 "Quest"/Block baselines in this harness use their scoring formulas + the same head-mean + shared-selected-set convention as this project, so the internal comparison is fair along this axis, and externally it should be written as "Quest-style scoring".

![](figures/bsa_report/fig05_head_aggregation_toy.png)

*Figure 5: A toy of head aggregation (the numbers are illustrative; from `2026-09-01_codesign_report_mixmax_region_aware.zh.md` §4.2). g4 is a retrieval-type head: the task says "search" → it responds strongly to the search box (3.2); g1 responds to layout / prominent text (banner 2.6). head-MEAN dilutes E3's 3.2 to 1.08, below all the navigation links (1.15), so the needle drops out when the budget is small; head-MAX keeps it, but also lifts the banner to 2.6.*

**Cost and boundaries (measured)**:

- B ≥ 3k: head-mean and head-max are indistinguishable end-to-end — on fixed-16, mixmax (head-mean) 96 vs Quest (head-max) 98 vs Block 99 vs dense 97 (n = 190, none of the paired comparisons significant). Even when diluted by averaging, the needle still lands safely inside the budget.
- B ≤ 2k: there is a real cost. On the 23 failure-enriched dumps, fixed-16 Block (head-max) needle 0.537 > tree head-mean 0.479; moving Block-style head-max onto tree units reaches at most 0.515.
- Independent corroboration: LessIsMore (arXiv 2508.07101) measured on reasoning models that the ground-truth top-k tokens of the individual heads overlap heavily, and on that basis replaced per-head selection with a unified selected set across heads; its GQA ablation shows unified set > independent per-head.
- The variant that switches only the DOM element chunks to head-max (the co-design of §5.5) is a net negative online, so all deployed configurations use head-mean.

### 3.4 Component 4: mixed granularity and budget allocation

Component 1 determines the chunk size of each region; this component determines how the budget is divided between regions.

**admission** (`python/tree_sparse_selector.py:328-375` `_select_by_budget_or_topk`; sglang side `bsa_sglang/selection.py:179-201` `token_budget_chunk_mask`):

1. All chunks (6-token element chunks and 256-token instruction chunks) enter **one and the same** mixmax_wn ranking — comparability across sizes depends entirely on the normalization of §3.2;
2. **system-prompt floor** (`TSA_SYS_FLOOR = f`): first, among the chunks before the first user turn, fill f·B by score (`:344-359`); any unused remainder is returned to the global pool;
3. The rest of the budget is filled greedily by the global ranking until the cumulative token count ≥ B (`:360-362`);
4. always-include as before.

**Hyperparameters**:

| Hyperparameter | Meaning | Values tried | Chosen | Basis |
|---|---|---|---|---|
| α | width normalization exponent | 0 / 0.25 / 0.5 | **0.25** | 0 is significantly worse (89 vs 96); 0.5 over-penalizes long chunks; measured β = 0.245 |
| f | system-prompt floor | 0.05 / 0.25 / 0.35 | **0.25** | online 50 / **57** / 56; 0.25 and 0.35 differ by 1 step, the cheaper one is taken |
| DOM min/max | merge lower limit / split upper limit for element chunks | — | **6 / 64** | offline needle sweep: too small a min explodes the unit count; max ensures long element lines are not merged into neighbouring lines |
| instruction-region min/max | merge lower limit / split upper limit for coarse chunks | — | **64 / 256** | offline mass-coverage sweep; 256 aligns with the kernel page |
| page_size | GPU-memory page | 16 / 64 | **16** | small pages make the chunk→page mapping read fewer irrelevant tokens (4506 actually read at B4096 vs 5032 with page 64) |

**One complete admission (toy, B = 128, always-include not counted)**: instruction-region coarse chunks C1 (rules, 80 tokens, score 1.40) and C2 (task, 30 tokens, 1.50); DOM element chunks E1 banner (12, 1.00), E3 search box (6, 1.08), and 8 navigation links N1–N8 (6 each, about 1.15). Floor 0.25 → 32 tokens: C2 gets in, the remaining 2 are returned. The global pool of 98 tokens is filled by score: C1 (80) gets in, 18 remain; N1, N2, N3 get in, 0 remain — E3 is out. This is the pattern that recurs in §5.3 (layout-responsive chunks and long instruction chunks crowd out the only element that carries a retrieval signal), and it is also where the cost of head-mean at B ≤ 2k lies; the region-aware solution is not to change the aggregation but to let E3 and the N chunks take part in the same ranking as equal 6-token chunks, and to use the floor so that coverage of the C region does not have to come at the expense of the DOM.

### 3.5 Engineering implementation and efficiency

- **CUDA selector**: fp8 envelopes are computed at prefill; at decode, a scoring kernel + fused top-k + page bitset (`csrc/ts_fused_page_select.cu`); the chunk → page mapping is a union of intervals (`selector :1242-1298`). A selected 10-token chunk pulls in the whole page it sits in, and a chunk that crosses a page boundary pulls in two pages, so "k chunks" is equal neither to k pages nor to a fixed number of tokens — the honest accounting for speed is page count × page_size, and the KV-read amounts in this report are counted that way.
- **Fix for the 1024-page capacity bug**: the original kernel's shared-memory page bitset held only 1024 pages and its marking loop had no bounds check, so with page 16 any request > 16.4k tokens produced an illegal memory access; it has been extended to 4096 pages with a clamp (`ts_tree_sparse.cu`, `ts_fused_page_select.cu`; reproducible with any `select_and_build_indices` call with `seq_len > 16384`).
- **Cross-step reuse** (LServe's idea, `TSA_RESELECT_K = k`, selector `:909-913` + `:1255-1298`): rescore only every k decode steps, and in between rebuild the pages from the cached chunk set (the sink/recent/generated windows are recomputed for the current seq_len). The Jaccard between the selected sets of adjacent steps is 0.81, and 0.71 at a distance of 8 steps; at k = 8, agree 55 vs 58 (p ≈ 0.63, not significant), PyTorch-path wall time −21%. Not used on the CUDA path (the fused reselection at 54 µs is already cheaper than the python rebuild).
- **sglang backend** (`bsa_sglang/backend_v2.py`, `BSA2_*` environment variables): the same chunker and scoring; the greedy packing of the token budget is implemented with pure device ops (no host sync, graph-safe); the region floor is unavailable under CUDA-graph capture (requires `--disable-cuda-graph`).
- **Limitations**: the CUDA mixmax kernel for batch > 1 is not implemented (routed to the per-request path); token budget / floor / region currently exist only on the PyTorch scoring path (the CUDA path supports only chunk-count top-k), so all end-to-end numbers in this report carry the time overhead of the PyTorch path (the same for every configuration, so the paired comparisons are unaffected).

![](figures/bsa_report/fig15_select_latency.png)

*Figure 15: The selection micro-benchmark (GB10, 200 selects, the token streams of real prompts + synthetic K/q; `bench_select.py` / `bench_select2.py`). (a) CUDA path: region-aware 51.7–65.0 µs vs fixed-16 envelope 74.1–114.0 µs, 1.4–1.75×; the unit count is 5–7× smaller but the kernel has a fixed launch overhead, so the speedup is smaller than the unit-count ratio. (b) The CUDA path is 24–54× faster than the PyTorch reference path; reuse every 8 steps reduces the overhead of the PyTorch path by 2–4.8×.*

---

## 4. Experimental setup

**Serving**: `serve.py` (OpenAI-compatible), temperature 0, xgrammar constraining the output to the JSON schema (without the constraint, sparse degenerates into malformed JSON), `--disable-cuda-graph` (the MoE grouped_mm conflicts with CUDA graphs on GB10). The full-attention baseline = the same code path with `top_k = 100000` (the selector selects all chunks, the dense bypass at `selector :896-907`).

**Data**: WebVoyager (643 tasks, 15 real websites) + GAIA-web (90 tasks). The reference trajectories were produced by a multimodal agent (Qwen3.5-omni) in a real browser; only successful tasks are kept.
- **half1**: 50 tasks / 332 steps / **190 index-action steps** (stratified sampling by category × length);
- **strat20**: a subset of half1, 20 tasks / 15 websites / **132 index steps**;
- **Mechanism dumps**: 15 steps (8 selection-sensitive + 4 controls + 3 Wolfram) and 23 steps (13 B2048 failures/successes + 10 B4096 disagreements), with the K of all layers, the q of every decode step, and the actual selected set of every layer (about 10–16 GB).

**Metrics**: agree / valid / none / fidelity (§Glossary); needle hit rate, mass recall (offline); paired sign tests and bootstrap 95% intervals. Re-running the same configuration at n = 190 changes agree by about 1 step; differences between configurations are only interpretable at ≥ 3–5 steps.

**Budget alignment**: fixed-16 uses page 16 × top-k (k = 128 / 256 corresponds to B = 2048 / 4096); tree uses a token budget. The actual amount of KV read by the two (page count × 16) is 4096 vs 4506 at B4096 (+10%, the cost of the chunk → page mapping), and 2048 vs 2341 at B2048.

**Methodological accidents (fixed; they affect historical conclusions)**:
1. **KV layout bug** (`2026-08-23_scoring_function_study.zh.md` §3): the original TSA indexed FlashInfer's page-interleaved one-dimensional KV buffer (`kv_cache.py:69` `elems_per_page = 2 * page_size * H_kv * D`, each page holding the K block first and then the V block) as if it were a row layout of `[tokens, H_kv, D]` (`ts_tree_sparse.cu:81-82` `k_idx = physical_loc*H*D + head*D + d`), so for tokens after page 0, half of the reads landed on values and half on the keys of the wrong page. Consequence: all numbers based on the default TSA path before 2026-08-21 (June's top-k 32 online success rate of 0.41% vs dense 69.7%, the early-August 3-way report) were selections made on noisy summaries and are all void. The same scoring mathematics, moved to the fixed path, took agree on the same batch of steps from 13.8% → 58.6%. `main` now uses separate K/V pools, with an assertion at the entry of `compute_centroids` (selector `:581-591`).
2. **The 1024-page capacity bug** (§3.5).
3. **Misconfigured fixed-16 baseline at B1024**: the first measurement used flat mode with the default `max-chunk-size 256`, so the chunk count (about 30) < top-k 64 → everything was actually selected = full attention. After correction, 55. Lesson: every baseline must be checked for the amount of KV it actually reads.

---

## 5. Results

### 5.1 B4096: all faithful scoring functions are indistinguishable from full attention

![](figures/bsa_report/fig09_e2e_B4096.png)

*Figure 9: agree on the 190 index steps of half1 at B = 4096 tokens (`2026-08-23_scoring_function_study.zh.md` §9 Stage C).*

- On fixed-16, the three faithful scoring functions (Quest, Block, mixmax) and dense are packed within ±2 steps; Quest and Block produce verbatim-identical outputs on 144/190 steps, and the disagreements ≈ repeat noise. This is the **transparent regime**: when the budget is large enough and the pages fine enough, the scoring function is invisible in the results.
- In the tree geometry, method 1 (merge ≥ 16 + mixmax_wn + token budget) is the only configuration that ties (96, fidelity 79%, valid 156 — the highest of all); un-normalized tree + Quest is significantly worse (89, p = 0.02).
- Why a per-layer needle hit of 0.65–0.90 is enough: page selection is independent per layer, and the model only needs to get the evidence in some of the layers; the dump review of the 10 disagreeing steps at B4096 shows that of the 5 "true losses", 3 become correct simply by re-running the same configuration, and in the rest the needle is selected in 57–99% of the decode steps and the wrongly chosen element is in the same or an adjacent chunk as the needle — a decision error, not a selection error.

### 5.2 The budget dimension: method 1 collapses between 2k and 3k, and region-aware repairs it

![](figures/bsa_report/fig10_budget_online.png)

*Figure 10: agree as a function of budget on the 132 index steps of strat20.*

- fixed-16 is almost lossless from B4096 to B1024 (59 → 55 → 55): the breadth of 16-token pages makes it robust at small budgets.
- Method 1 equals dense at B3072 (57) and falls to 48 at B2048 (fidelity 57% vs fixed-16's 63–66%, true losses 15 vs 10) — the collapse boundary lies between 2k and 3k, roughly the combined coverage requirement of the high-mass part of the system prompt and the DOM needle region.
- Method 2 (region-aware + floor 0.25) equals dense at B2048 (57), and at B1024 gets 53 vs the baseline's 55 (paired 13 wins 14 losses, not significant), with valid 123 vs 100 and none 8 vs 31 — at B1024 the loss mode changes from "format breakdown" to "element discrimination".

### 5.3 The two failure mechanisms at small budgets and their separate fixes

![](figures/bsa_report/fig12_mechanisms_B2048.png)

*Figure 12: Dump review of the 13 failure/success steps at B2048 (`case_discord2048_B2048.md`). (a) When the needle sits in a 109–219-token `<form>`/`<div>` subtree, method 1 almost never selects it (0.01–0.20), while fixed-16 pages hit the same needle at 0.15–0.66; small-chunk needles (23–41 tokens) do not have this problem. (b) At B2048, method 1 covers only 7–14% of the system-prompt tokens (36–62% at B4096), consistent with the two none cases that still had not produced an action after 1023 decode steps.*

The two mechanisms conflict with each other at a fixed budget: guaranteeing the system prompt takes budget away from the DOM and worsens mechanism A; splitting large chunks lets fragments compete with small needles for budget. Three attempts to "change only the selection" (sub-block hierarchical scoring, tree + fixed-16 mixed allocation, per-window quotas) all failed offline. The solution is to hand the two mechanisms to two different components:

![](figures/bsa_report/fig13_ablation_floor_chunking.png)

*Figure 13: B2048, the 132 steps of strat20. Adding only floor 0.05 without changing the chunking: agree 48 → 50, none 40 → 26 (mechanism B fixed, mechanism A still present); then switching to region-aware chunking: 57, none 19 (below full attention's 33).*

![](figures/bsa_report/fig11_method_evolution.png)

*Figure 11: Method 1 → method 2 compared under the same setting on the two evaluation sets (the only variable is the per-region chunking parameters + floor). strat20 +9 (17 wins 8 losses, p = 0.108), half1 +12 (29 wins 17 losses, p = 0.104); method 2's +2 / +3 relative to fixed-16 is within run-to-run variance. The method 1 row on half1 is tree + Quest-style scoring with chunk count k = 32 (≈ 2k tokens).*

### 5.4 The real limits of efficiency and end-to-end speed

1/3–1/7 as many scoring units and 1.4–1.75× faster selection (Figures 14 and 15) are reproducible. End-to-end decode speed is a different matter:

![](figures/bsa_report/fig16_speed_arithmetic.png)

*Figure 16: (a) The early-August breakdown of a decode step for 30B-A3B on GB10 with a 29.5k-token prompt (`2026-08-07_speed_study_tsa_vs_quest_vs_block.en.md` §8): the MoE expert GEMMs take 50%, and paged attention reading the full 29.5k KV takes only ≥ 13 ms (8.5%); under full attention, going from a 25k-character prompt to a 118k-character prompt changes the step time from 134.9 → 134.5 ms, i.e. no change. By Amdahl's law, the ceiling for perfect sparsification is 1.09× (1.43× on an ideal machine). (b) The upstream authors' measurement on B200 / Qwen3-VL-8B (dense MHA) / 9.6k prompt (README): sparse + CUDA graph against SGLang full attention, 1.74× (bs = 1) to 3.53× (bs = 64).*

Conclusion: the value of sparse attention depends on the share of attention within a decode step. GQA 8:1 cuts KV traffic to 1/5 of dense MHA, and MoE cuts weight reads by 1/2 or more; together they leave attention at only about 8–15% of a decode step on 30B-A3B. For attention to become the dominant term, one needs a dense MHA model, a context of ≥ 60k tokens, or a large batch (KV traffic grows linearly with batch while weight reads stay constant). The accuracy conclusions of this project are unaffected by this, but any "faster" claim must be restricted to selection overhead and KV bytes read.

### 5.5 Refuted directions (on record, to avoid redoing them)

- **hierarchical selection** (branch-and-bound descent over the DOM tree): offline needle 0.544 vs 0.547 for flat, 170 vs 112 scoring operations — no more accurate and more expensive; the tree of a web prompt is shallow and wide, and the number of nodes at the starting level of the descent is already close to the flat total. The applicable scenario would be full pages of ≥ 50k tokens.
- **sub-block scoring** (LServe-style, max over 16-token sub-blocks), **tree + fixed-16 mixed allocation**, **per-window quotas**, **keeping only the first sub-block of a chunk**: at B2048 offline, none is higher than whole-chunk selection, or they read 26% more KV.
- **Adding only sys-floor 0.5 without changing the chunking**: none 40 → 26 but agree only +2, and fidelity drops to 49%.
- **co-design (method 4)** = method 2 + a hard three-region partition (sys 0.25 / DOM 0.55 / rest 0.20) + priority admission of interactive elements in the DOM region + head-max for the DOM element chunks:

![](figures/bsa_report/fig17_codesign_offline_vs_online.png)

*Figure 17: (a) On the 23 failure-enriched dumps, method 4's offline needle hit is 40% / 57% higher than fixed-16 Block; (b) online, however, it loses to the baseline by 9 steps at B1024 (paired vs method 2: net −6, a true accuracy deficit), and at B2048 it ties method 2 on the clean steps but 17/132 steps do not finish within 1800 s (per-head scoring stacked on top of the PyTorch path).*

Methodological lesson: a failure-enriched offline proxy ranks only on the hard cases, whereas online agree is determined by the whole distribution; offline proxies are used only for relative ranking within a family of configurations and for mechanism diagnosis, and any verdict across families (vs the baseline) always waits for online results. Another ablation confirmed the reasoning that "the instruction region must not use head-max": switching the instruction region to a 32-head max reduced system-prompt coverage by 15% to 17% with the needle unchanged — the bound of a long chunk is loose, and taking the max over many heads picks the largest out of "one loose-bound sample per head", which amplifies the noise.

---

## 6. Summary of differences from conventional sparse attention

| Design axis | Conventional (Quest / BlockSparse / LServe) | This project | Why |
|---|---|---|---|
| Selection unit | fixed 16-token page, content-agnostic | variable-length semantic chunk, boundaries aligned to DOM elements and ChatML segments | the element line is the minimal unit of the action target; 3–5× fewer units |
| Unit size | uniform across the whole text | **per region**: DOM 6–64, instructions/history 64–256 | the instruction region only needs coverage, the DOM region needs granularity |
| Summary | envelope or centroid | envelope (fp8) | needle-sensitive; the centroid suffers dilution on variable-length chunks |
| Scoring | Quest bound / centroid softmax | Quest bound + **width term × (16/L)^0.25** | under variable-length units the bound inflates with length; normalization is the enabling condition |
| Head handling | each head selects its own (Quest) / per-head softmax then max (Block) | mean within the GQA group + mean across KV heads, shared selected set | a single page table, 8× less compute; no cost at B ≥ 3k |
| Budget accounting | page count | token count (greedy packing) + **region floor** | under variable-length units, page count / chunk count / token count disagree; the floor preserves instruction coverage |
| Selection frequency | every step, every layer (LServe can reuse) | every step, every layer, optional reuse every 8 steps | same as LServe |
| Applicable budget | robust at B ≥ 1k (fine-grained breadth) | on par with dense at B ≥ 2k; ties fixed-16 at B1024 with better format compliance | at small budgets, the needle inside a large chunk gets starved — a structural cost |
| End-to-end speed | depends on the model | same; the selection itself is 1.4–1.75× faster | the share of attention sets the ceiling (Figure 16) |

---

## 7. Contributions, boundaries, and next steps

**Claimable (by strength of evidence)**:
1. **Chunking and budget allocation partitioned by the structure of the web-agent prompt** (element-granularity DOM region / coarse-granularity instruction region / unified normalized ranking / budget floor). Existing budget-allocation work operates along the head dimension (Ada-KV) and the layer dimension (PyramidKV); no precedent has been found for allocation by semantic region of the prompt. Evidence: +9 / +12 under the same setting vs single-parameter chunking, full attention reached at B2048, 1/3–1/5 as many units, selection 1.4–1.75× faster.
2. **Length normalization of upper bound scoring on variable-length chunks** (correcting only the width term; α fitted from data and independently verified by geometric measurement).
3. **Quantitative characterization of the failure mechanisms and the applicable range**: the two mechanisms (granularity × instruction coverage) and their respective fixing components; the three budget ranges (≥ 3k saturated / ≈ 3k transition / ≤ 2k needs breadth); the refutation of hierarchical and co-design; the conditions for using offline proxies.
4. **Engineering artifacts**: the CUDA mixmax kernel (Jaccard 0.99–1.00 with the reference implementation), two upstream bug fixes (with reproducers), the applicable range of cross-step reuse, the sglang backend.

**Boundaries**: one model (30B-A3B, GQA 8), one workload (teacher-forced replay of browser trajectories), paired statistical power at n = 132–190; whether head-mean is free at other GQA widths / on other tasks has not been tested; the **online task success rate** after the bug fixes has not been re-run (all earlier online data is void); the fixed-64/32 same-budget comparison has not been measured; region and floor exist only on the PyTorch scoring path, and the token budget on the CUDA path remains to be ported.

**Next steps (ordered by leverage)**:
- **Reuse of chunk identity across turns**: the web agent re-prefills 6–14k tokens at every step, and prefill is the dominant latency term; byte-exact prefix caching covers only about 59%, and the rest are DOM subtrees whose "content is unchanged but position has shifted". Semantic chunks have an identity across steps (fixed pages do not), which allows transfer of selection priors or chunk-granularity position-repair reuse. KV byte reuse is already crowded (CacheBlend / EPIC / KVShare); reuse of selector state and identity preservation of DOM elements across page modifications are unoccupied.
- Port token budget + region floor into the CUDA selector; a batched mixmax kernel.
- Hierarchical selection on full DOM pages of ≥ 50k tokens; the generality of length normalization on other structured inputs (code ASTs, markdown sections).

---

## Appendix A: Code map (spark00 `/home/shiqihe/workspace/TreeSparseAttention`, branch `shiqihe/region-aware` @ `5afd993`)

| Mechanism | Location |
|---|---|
| Per-layer, per-step selection call | `models/tree_sparse_patch.py:531-543`; `models/direct_decode.py:1194-1198`; the CUDA-graph lagged-query pre-pass `direct_decode.py:813-830` |
| Summary computation at prefill (envelope / centroid, fp8) | `python/tree_sparse_selector.py:562-641` |
| Tree parsing | `python/tree_parser.py:90` `parse_webarena_tree`; `:269` `parse_chatml_tree` |
| Subtree chunking / region-aware chunking | `python/tree_parser.py:880-981` / `:1015-1069`; DOM span `:998-1012`; user turn `:984-995` |
| The selector's region-chunking entry and parameters | `python/tree_sparse_selector.py:1307-1325`; environment variables `:195-218` |
| mixmax / mixmax_wn scoring (PyTorch) | `python/tree_sparse_selector.py:1327-1347`; head group mean `:1153` |
| mixmax CUDA kernel | `csrc/ts_tree_sparse.cu:402` `score_fp8_mixmax_kernel`, launcher `:1191` |
| Token-budget greedy admission + sys floor | `python/tree_sparse_selector.py:328-375` |
| always-include and chunk → page | `python/tree_sparse_selector.py:1349-1409`; fast page union `:1242-1298` |
| Cross-step reuse | `python/tree_sparse_selector.py:909-913` (`TSA_RESELECT_K`) |
| sglang-side scoring / budget / page projection | `bsa_sglang/selection.py:66-93`, `:179-201`, `:204-243`; backend `bsa_sglang/backend_v2.py` (`BSA2_*`); config `bsa_sglang/config.py` |
| Unit tests | `bsa_sglang/test_selection_scoring.py` (2-corner regression, mixmax closed form, bound properties, normalization), `bsa_sglang/test_region_floor.py` (two-phase floor, chunk tiling), `tests/test_cuda_mixmax_ab.py` (CUDA vs torch) |
| Reproducer for the layout bug | `simulator/runs/sparse3way-20260721/scoring_case_study/test_cuda_layout_realpath.py` |

## Appendix B: Reproduction commands

```bash
# standalone server, validated region-aware config (B2048)
TSA_HYBRID_GEOM=1 TSA_BUDGET_TOKENS=2048 TSA_SYS_FLOOR=0.25 \
python3 serve.py --model-path /models/Qwen3-VL-30B-A3B-Instruct \
  --host 0.0.0.0 --port 10000 --tree-parse-mode webarena \
  --page-size 16 --top-k 64 --scoring-method mixmax_wn \
  --max-decode-tokens 1024 --max-batch-size 4 --disable-cuda-graph
```

```bash
# fixed-16 Quest-style baseline at the same budget
python3 serve.py --model-path /models/Qwen3-VL-30B-A3B-Instruct \
  --host 0.0.0.0 --port 10000 --tree-parse-mode fixed \
  --page-size 16 --top-k 128 --scoring-method envelope \
  --max-decode-tokens 1024 --max-batch-size 4 --disable-cuda-graph
```

```bash
# full-attention reference: identical command with --top-k 100000
# offline replay (container /workspace/sparse3way): python3 traj_eval.py --tag <TAG> --data offline_h1_strat20_idx.jsonl --conc 6
# figures: python simulator/figures/bsa_report/make_figures.py
```

## Appendix C: Index of data and results

| Content | Path |
|---|---|
| Evaluation sets | `simulator/runs/sparse3way-20260721/offline_half1.jsonl` (332 steps), `offline_h1_strat20.jsonl` (209 steps / 132 idx), `sample_100.json` |
| Per-step results for every configuration | `simulator/runs/sparse3way-20260721/result_*.jsonl`, `partial_*.jsonl` (the full set is at spark00 `/workspace/sparse3way/`) |
| Paired analysis scripts | `analyze_stageB.py` / `analyze_stageC.py` / `analyze_stageD.py` (same directory) |
| Offline mechanism sweeps | `scoring_case_study/analysis/sweep_v3.md`, `sweep_wn.md`, `sweep_explore.md`, `summary_v3.json` |
| Dump reviews | `case_dumps/case_discord10_B4096.md`, `case_discord2048_B2048.md`; spark00 `/workspace/scoring_case_study/dumps_*` |
| Chunk-by-chunk rendering of the five chunking schemes on the same prompt | `simulator/CHUNKED_{region_new,mixmax_RA,mixmax_noRA,main_subtree,V2_cap32,SUBTREE_FULL}.md`, compared in `2026-09-02_chunking_comparison.zh.md` |
| Speed study | `simulator/report.md` §8–§9; selection micro-benchmark spark00 `/workspace/scoring_case_study/bench_select*.py`, `analysis/bench_select.jsonl` |
