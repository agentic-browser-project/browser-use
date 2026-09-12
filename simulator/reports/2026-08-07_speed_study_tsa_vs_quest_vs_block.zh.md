# Web Agent 的 Sparse Attention：TSA vs Quest vs BlockSparse

**完整报告，不假设任何背景知识。**（英文版：`2026-08-07_speed_study_tsa_vs_quest_vs_block.en.md`）

被测模型：`Qwen3-VL-30B-A3B-Instruct`。硬件：NVIDIA GB10（sm_121），119 GB unified memory。
工作负载：WebVoyager + GAIA 浏览器 agent 轨迹。
时间：2026-08-03 → 2026-08-07。除非注明引用，所有数字均在此窗口内实测。

---

## 0. 结论速览

1. **Sparse attention 在这个工作负载上损失 accuracy，且换不到任何可测量的速度。**
2. **换不到速度的原因不是稀疏代码有 bug，而是 attention 本身只占一个 decode step 的约 8.5%。**
   完美稀疏的理论最优是 **1.09x** 端到端加速 —— 比测量噪声还小。
3. **真正的瓶颈是一个被 HuggingFace 静默默认选中的 Mixture-of-Experts (MoE) kernel。**
   `experts_implementation="grouped_mm"` 占了 153.5 ms decode step 中的 **77.1 ms**（一半），
   且比自己的 memory-bandwidth roofline 慢 **4.7 倍**。microbenchmark 显示换成 `batched_mm` 省约
   33 ms。*一个 flag 的收益超过整个算法* —— **但这个收益尚未在 end-to-end 层面被确认**（见 §8.6）。
4. **Batch size 是 attention 占比的最强杠杆（§9.3）** —— B=1 时 attention 只占 32%，B=32 时达 64%，超过 Quest 论文中 Llama-2-7B 的 55%。本报告先前称 "batch 不是杠杆"，那是错的：KV 流量精确线性于 B，而权重流量次线性且在 128 个 expert 处饱和。**所有速度实验都跑在 B=1 上，即该方法最不可能获胜的设置。**
5. **精度上，budget 才是决定因素，而且排序会随 budget 翻转。** budget 4096 时 TSA 显著优于两个
   baseline；8192 时两个 baseline 与 full attention 已无统计差异，而 TSA 仍显著落后；
   16384 时 TSA 精确追平 full attention。
6. **我自己早期的若干测量是错的，本报告逐条撤回**（§5）。它们的共同特征是**静默失败** ——
   产生看似合理的数字而非报错。这就是 §5 存在的理由。

---

## 1. 要解决什么问题

### 1.1 Web agent

"Web agent" 指的是驱动真实浏览器的 LLM。每一步，agent 收到当前页面的文本描述 —— 每个可点击元素
都带一个数字 **element index** —— 然后必须回复类似 `click(index=41827)` 的动作。

这些页面描述很大。本数据集中在 **25,000 到 118,000 字符**（约 6,000 到 29,500 tokens），
且随步数累积。长上下文在这里不是可选项，是任务的本质。

### 1.2 长上下文为什么昂贵

Transformer 维护一份 **KV cache**：每 token 每 layer 存两个向量（一个 key、一个 value），
以便后续 token 能 attend 到之前的内容。生成**一个** token 就要读**整份** KV cache。
上下文翻倍，读取量翻倍。

### 1.3 Sparse attention 的提议

KV cache 中的绝大部分与任一给定 query 无关。**Sparse attention** 只挑一小部分 —— 一个
**budget** —— 只读这部分。如果挑得好，就能只读 2,048 个 token 而非 29,500 个，模型表现几乎不变。

由此引出两个问题，本项目都做了测量：

- **Accuracy**：模型是否仍能选对 element index？
- **Speed**：少读 KV 是否真的让生成变快？

答案分别是"不，会退化"和"不，不会"—— 而第二个答案的**原因**是本报告最有价值的结果。

---

## 2. 三种方法

三者都把 KV cache 切成 **chunk**，对每个 chunk 相对当前 query 打分，保留分数最高的若干 chunk，
只 attend 这些。它们的差异恰好只有两个设计选择。

### 2.1 设计选择 A —— 如何切分上下文

**Fixed chunking（Quest、BlockSparse）**：每 16 个 token 切一刀。简单、均匀、无视结构。

**Variable / tree chunking（TSA，本项目在研的方法）**：按**文档结构**切。浏览器页面有天然边界 ——
一个元素、一条消息、一个区块 —— TSA 的 parser 输出 16 到 256 token 不等、对齐到这些边界的 chunk
（`--tree-parse-mode webarena`）。

*为什么这可能重要。* 打分会把一个 chunk 压缩成一个 summary vector。chunk 越大，
单个有辨识度的 token 越容易被邻居稀释。本项目早期实测正是如此：1–8 token 的 chunk 有 84.4% 被选中，
129+ token 的只有 8.4%。结构对齐的 chunk 把语义连贯的东西放在一起，summary 就没那么失真。

### 2.2 设计选择 B —— 如何给 chunk 打分

**Centroid scoring（BlockSparse）**：把 chunk 内的 key vector 取平均成一个 mean vector，
再与 query 做点积。*失效模式：* 平均会相互抵消。一个强烈指向 query 的 key，
和十五个指向别处的 key 平均后，得到一个平庸的均值 —— 于是含有答案的那个 chunk 分数很低。

通俗说法：要判断一本书值不值得读，你去读这本书所有句子的平均。
一本只有一句关键的书，平均下来就是噪声。

**Envelope scoring（Quest、TSA）**：保留 chunk 内每个维度的 **min 和 max**，
然后给出该 chunk 内任意 key 可能达到的**上界**：query 为正的维度取 max，为负的维度取 min。
这是一个 admissible over-estimate —— 它绝不会低估一个含有匹配项的 chunk。

通俗说法：不看平均，而是问"这本书里**最好**的那一句，乐观估计有多好？"
只有一句关键的书现在分数就很高。

### 2.3 三个配置

| 方法 | Chunking | Scoring | 启动参数 |
|---|---|---|---|
| **TSA (tree)** | 变长、结构对齐 | envelope | `--page-size 64 --tree-parse-mode webarena --scoring-method envelope` |
| **Quest** | 固定 16 tok | envelope | `--page-size 16 --tree-parse-mode fixed --scoring-method envelope` |
| **BlockSparse** | 固定 16 tok | centroid | `--page-size 16 --tree-parse-mode fixed --scoring-method centroid` |

TSA 与 Quest 只差 chunking；Quest 与 BlockSparse 只差 scoring。**这个对比每次只隔离一个变量。**
这正是设计的用意。

### 2.4 公平性控制：matched budget

单看 `top_k` 跨方法是没有意义的，因为 page size 相差 4 倍。必须固定的是
**budget = page_size × top_k**，即每个 decode step 实际 attend 的 KV token 数。

| Budget | TSA | Quest / BlockSparse |
|---|---|---|
| 2048 | page 64 × k 32 | page 16 × k 128 |
| 4096 | page 64 × k 64 | page 16 × k 256 |
| 8192 | page 64 × k 128 | page 16 × k 512 |
| 16384 | page 64 × k 256 | page 16 × k 1024 |

参照系：在 118k 字符（约 29.5k tokens）下，4096 的 budget 约读取上下文的 **1/7**。

---

## 3. System implementation

### 3.1 模型

| 属性 | 数值 | 为什么在这里重要 |
|---|---|---|
| Layers | 48 | 任何 per-layer 开销每 token 都要付 48 次 |
| Experts | 128 个，每 token **激活 8 个** | 总参数 30B 但只激活约 3B —— 权重读取仍然主导 |
| Hidden size | 2048 | |
| Attention heads | 32 query / **4 KV**（GQA 8:1） | **KV cache 比 MHA 小 8 倍** —— 见 §9 |
| Head dim | **128** | 不是 `hidden/heads = 64`。算错这个让我的 KV 估计少了一半（§5.4） |
| 每 token 的 KV | 2 × 4 × 128 × 2 B × 48 layers = **96 KiB** | |
| 每步读取的激活权重 | **6.083 GB** | |
| dtype | bfloat16 | |

### 3.2 Serving stack

不是 vLLM 也不是 SGLang —— 是一个专门写的 `serve.py`（1,251 行），配 FlashInfer paged KV。
三种方法跑**同一个二进制**，只有启动参数不同。这是优点（没有跨框架混淆）也是缺点
（没有框架级优化可以依赖）。

每个 decode step、每个 layer，sparse path 依次执行：group query heads → 把 query 量化到 fp8 →
对所有 chunk 打分 → fused page-select → top-k → 构建 page indices。
每步大约多出 250–290 次 kernel launch。

### 3.3 实测硬件

- 有效内存读带宽：**222.7 GB/s**（在 512 MB bf16 reduction 上实测；标称 273）
- Unified memory：119 GB，host 与 device 共享

---

## 4. 实验设计

### 4.1 数据

从 WebVoyager + GAIA 采样 100 个 task，全部是参考模型（`Qwen3.5-Omni`，dense attention）
成功完成的，且每个站点类别至少一个。切成两半，每半 50 条**完整轨迹** ——
每个 task 的每一步都在，绝不做 step 级采样，因为不完整的轨迹无法做 end-to-end 评估。

以下所有结果使用 **half1：50 tasks / 332 steps / 190 个产出 index 的 step / 142 个其他 step**。

### 4.2 指标

对每一步，我们回放完全相同的参考上下文，再比较模型的动作：

- **`valid`** —— 所选 element index 在页面上确实存在。
- **`agree`** —— 所选 index 与参考轨迹的 index 完全一致。
- **`none`** —— 模型根本没有产出 index。
- 对非 index step（scroll、done 等），比较 **action type** 是否一致。

**`agree` 是一个严格的 proxy，不是 task success。** 很多 task 有多条有效路径；
点了另一个同样正确的按钮会被算作不一致。§7.2 用 full attention 作为标尺量化了它到底有多严格。

### 4.3 统计方法

step 是**嵌套在 task 内**的，把 190 个 step 当作独立样本会高估显著性。
因此下文所有检验都是 **cluster-level**，以 *task* 为单位：对"哪个配置在该 task 上答对更多 index
step"做 sign test，外加 cluster bootstrap（重采样 task，重算汇总 agree-rate 之差）。

---

## 5. 已发现并修复的方法学失败

之所以记录，是因为**每一个都产生了看似合理的数字而非报错**。三轮结果因此作废。

### 5.1 `ignore_eos` 完全没生效（静默，代价：一整轮 benchmark）

要跨配置比较 per-token latency，每个配置必须生成**相同数量**的 token。
我加了 `ignore_eos` 并做了验证：各 cell 返回恰好 128 token。**那次验证是巧合。**
浏览器 prompt 天然产出很长的 JSON，几乎每个 cell 都撞上了 `max_tokens` 上限，`ignore_eos` 无关紧要。

根因：`serve.py` 在**未提供 schema 时仍然套用内置 JSON grammar**，
当 JSON 对象闭合触发 `matcher.is_terminated()` 时解码就停止。这个 break 没有被 `ignore_eos` 管控。
用短 prompt 直接测试即暴露：开关打开是 21 token，关闭也是 21 token。
修复方式是在 benchmark 模式下彻底绕过 grammar。

### 5.2 两段相减的计时被 scheduler 摧毁（代价：一整轮）

decode 时间原本算作 `time(max_tokens=129) − time(max_tokens=1)`。
两次调用都排在同一个 150 ms batch-collect scheduler 后面，队列噪声会落进任意一项。
症状：4 个 cell 的 `decode_s ≤ 0`（报出 `2.56 × 10⁸ tok/s`），以及最关键的破绽 ——
**相同工作量的 prefill 跨配置相差 20 倍**（2.16 s vs 45.2 s），而 sparsity 只作用于 decode。

修复方式是扫 `max_tokens` 并拟合 `time = intercept + slope × tokens`。
intercept 就是 prefill，它**必须**跨配置一致 —— 一个内建的 validity check。修复后 prefill 跨度为 **1.08x**。

### 5.3 修好的估计量仍然分辨不出它要测的东西（代价：一整轮）

`--points 1,65,129` 是对称的，所以 OLS slope *恒等于* `(t₁₂₉ − t₁)/128`：
**中间那个点杠杆为零**，R² 测的只是它的一致性，与 slope 的精度无关。
配合每个 cell `n=1`，任何配置间差异的 **95% CI 是 ±20 ms** —— 比表里所有差异都宽。
当时报告的 per-config 速度排名是噪声。

在 `bench5.py` 中修复：5 个有杠杆的点、每点重复 3 次、并报告 slope 的**标准误**，
使差异可以被检验而不是靠眼看。

### 5.4 `head_dim` 被假设为 64，实际是 128

我按 `head_dim = hidden/num_heads = 2048/32 = 64` 计算。
服务器日志明确打印 `QO heads: 32, KV heads: 4, Head dim: 128`。
所有 KV 流量数字都**少了一半**。"attention 占 decode 时间不到 2%"这个说法就是它的后果；
正确的界是 **8.5%–15%**。

### 5.5 "CUDA graph 不可用"是 harness 假阴性

探针在**单次** `p=0` 读数时就判定死亡，没有去抖（而 `wait_ready` 要求连续三次）。
再加上 `docker restart` 后 `sleep 45` 太短，三次后端尝试都在 51–52 秒"死亡"，
其中两个日志文件是 **0 字节** —— 进程根本没启动过。
被判定死亡的那个配置后来被发现**正常服务中**。真正的答案见 §8.4。

### 5.6 `full` 不是 dense baseline

`tree_sparse_selector.py:165` 是 `selected_chunks = min(top_k_chunks, total_chunks)`。
设 `--top-k 100000` 时所有 chunk 都被选中 —— 但整条 selection pipeline 仍然完整运行。
`full` 这一臂是 *sparse-code-with-everything-selected*，估计比真正的 dense path 多花 **+2 到 +4 ms**。
这只有分辨率的五分之一，不改变任何结论，但意味着**速度实验里的每一个比值都是 sparse-vs-sparse**，
这一臂应当改名为 `topk=all`。

### 5.7 `traj_eval.py` 没有 resume

它用 `"w"` 模式打开 `partial_<tag>.jsonl`。每一次"自愈重启"都静默地从零重来。
结果仍然正确，但浪费了数小时。

---

## 6. Accuracy 结果

50 tasks / 332 steps / 190 个 index step。**`DENSE` = full attention，同样的回放，其余全部相同。**

| 配置 | Budget | `idx_valid` | `idx_agree` | 非 index step 类型一致 | task 全对 | 每 task 平均比例 |
|---|---|---|---|---|---|---|
| **DENSE** | full | **154/190 (81.1%)** | **97/190 (51.1%)** | 62/142 (44%) | 16/50 | 0.620 |
| tree_k64 | 4096 | 134/190 (70.5%) | 55/190 (28.9%) | 61/142 (43%) | 6/50 | 0.364 |
| quest_B4096 | 4096 | 132/190 (69.5%) | 39/190 (20.5%) | 45/142 (32%) | 8/50 | 0.312 |
| block_B4096 | 4096 | 136/190 (71.6%) | 35/190 (18.4%) | 50/142 (35%) | 5/50 | 0.248 |
| tree_k128 | 8192 | 151/190 (79.5%) | 83/190 (43.7%) | 58/142 (41%) | 12/50 | 0.524 |
| quest_B8192 | 8192 | 139/190 (73.2%) | 90/190 (47.4%) | 68/142 (48%) | **16/50** | 0.583 |
| block_B8192 | 8192 | 151/190 (79.5%) | 91/190 (47.9%) | 59/142 (42%) | 15/50 | 0.594 |
| **tree_k256** | **16384** | 151/190 (79.5%) | **97/190 (51.1%)** | — | — | **0.624** |

**`tree_k256` 精确复现了 full attention** —— 两者都是 97/190，每 task 平均比例 0.624 vs 0.620。

### 6.1 与 full attention 的距离

| 对比 | dense 更好的 task 数 | 更差 | p | agree-rate 差距 95% CI |
|---|---|---|---|---|
| vs tree_k64 (4096) | 25 | 0 | <0.0001 ✱ | +0.152 … +0.296 |
| vs quest_B4096 | 32 | 1 | <0.0001 ✱ | +0.229 … +0.383 |
| vs block_B4096 | 33 | 1 | <0.0001 ✱ | +0.240 … +0.414 |
| vs tree_k128 (8192) | 14 | 2 | **0.0042** ✱ | +0.032 … +0.120 |
| vs quest_B8192 (8192) | 13 | 8 | 0.383 ns | −0.023 … **+0.095** |
| vs block_B8192 (8192) | 9 | 6 | 0.607 ns | −0.026 … **+0.094** |
| **vs tree_k256 (16384)** | **5** | **5** | **1.000 ns** | **−0.033 … +0.033** |

**要看置信区间，不能只看 p 值。** 三行 "ns" 都未能拒绝原假设，但信息量完全不同。
Quest 和 BlockSparse 在 8192 的区间延伸到 +9.5 个百分点 —— 那是 *absence of evidence*，
在这个样本量下与"持平"到"存在实质差距"都相容。`tree_k256` 的区间是 ±3.3 个百分点：
**evidence of absence。只有 TSA 在 16384 下被真正证明等价于 full attention。**

### 6.2 同 budget 内的两两对比

| 对比 | sign test | p | 差值 95% CI |
|---|---|---|---|
| **tree_k64 vs quest_B4096** | 19 / 6 | **0.015** ✱ | +0.030 … +0.140 |
| **tree_k64 vs block_B4096** | 26 / 11 | **0.020** ✱ | +0.033 … +0.179 |
| quest_B4096 vs block_B4096 | 12 / 7 | 0.359 ns | −0.029 … +0.070 |
| tree_k128 vs quest_B8192 | 10 / 14 | 0.541 ns | −0.107 … +0.027 |
| tree_k128 vs block_B8192 | 7 / 13 | 0.263 ns | −0.114 … +0.028 |
| quest_B8192 vs block_B8192 | 8 / 8 | 1.000 ns | −0.050 … +0.040 |

### 6.3 失败模式拆解 —— 本报告最干净的因果结果

| 配置 | agree | 有效但不同 | **invalid（幻觉 index）** | 未产出 index |
|---|---|---|---|---|
| **DENSE** | 51% | 30% | **0%** | 19% |
| tree_k64 | 29% | 42% | 7% | 23% |
| quest_B4096 | 21% | 49% | 6% | 24% |
| block_B4096 | 18% | 53% | **10%** | 18% |
| tree_k128 | 44% | 36% | 3% | 18% |
| quest_B8192 | 47% | 26% | 6% | 21% |
| block_B8192 | 48% | 32% | **1%** | 19% |

**Full attention 在 190 步中幻觉出不存在的 element index 的次数是 0。**
每个 sparse 配置都会以 1–10% 的比例发生。归因是明确的：*凭空发明一个页面上不存在的元素，
是 sparsity 造成的。*

同样重要的是，**"未产出 index" 在所有配置（包括 dense）都是 18–24%。**
这个失败**不是** sparsity 造成的，不能算到它头上。

### 6.4 多步 task

在 38 个含 ≥2 个 index step 的 task 中，*每一个* index step 都答对的数量：

| | DENSE | tree_k64 | quest_B4096 | block_B4096 | tree_k128 | quest_B8192 | block_B8192 |
|---|---|---|---|---|---|---|---|
| 全部 agree | **5/38** | 0/38 | 0/38 | 0/38 | 2/38 | **6/38** | 4/38 |
| 全部 valid | **21/38** | 12/38 | 14/38 | 16/38 | 16/38 | 16/38 | **21/38** |

注意 dense 自己也只有 5/38。**这个指标主要在衡量它自身的严苛程度。**
若单步 agree 率为 `p`，一个 `k` 步 task 需要 `p^k`：在 dense 的 51% 下，5 步 task 是 3.5%。
要让一半的 5 步 task 全对，单步需要 **87%**；10 步需要 **93%**。这里没有任何配置接近，dense 也不例外。

---

## 7. Accuracy 结果的解释

### 7.1 排序随 budget 翻转，两侧都有机制解释

**在 4096，TSA 获胜（p = 0.015 / 0.020）。** 这正是该方法设计针对的区间。
budget 紧张时，*选哪些* chunk 起决定作用，结构对齐的变长 chunk 避开了 centroid dilution 的失效模式。
与此一致的是，`quest` vs `block` —— 同样 chunking、不同 scoring —— **不显著**（p = 0.359）。
**chunking 才是关键变量；一旦 chunking 固定，scoring 是二阶的。** 这直接支持了 TSA 的核心论点。

**在 8192，TSA 失去优势，且是唯一仍显著落后 dense 的方法（p = 0.0042）。**
这是令人不适的结果，budget-16384 的实验就是为解决它而加的。

### 7.1.1 budget 曲线交叉 —— TSA 能达到 full attention 水平，但来得晚

`idx_agree` 随 budget 的变化：

| Budget | TSA (tree) | Quest | BlockSparse |
|---|---|---|---|
| 4096 | **28.9%** | 20.5% | 18.4% |
| 8192 | 43.7% | **47.4%** | **47.9%** |
| 16384 | **51.1%** | 未跑 | 未跑 |
| *(full attention)* | *51.1%* | *51.1%* | *51.1%* |

有两件事在发生，机制并不相同。

**TSA 优雅降级；fixed chunking 有阈值。** 从 4096 到 8192，TSA 涨 14.8 个百分点，
Quest 涨 26.9，BlockSparse 涨 29.5。固定 16-token chunking 在紧 budget 下*灾难性*地差，
一旦 budget 越过某个阈值就*极好*。这是 **resolution effect** 的特征：
在 8192，约束从"有没有找到正确的区域"变成"能多精确地落在正确的 token 上"，
而 page-16 的分辨率是 page-64 的 4 倍。TSA 的 64-token page 买到了它不再需要的覆盖，
却付出了它现在需要的分辨率。

**TSA 是唯一被证明与 full attention 等价的方法。** 在 16384 它落在 97/190 ——
与 dense 完全相同的计数 —— 置信区间 ±3.3 个百分点，sign test 5/5。
Quest 和 BlockSparse 在 8192 *未被证明与 dense 不同*，但它们的区间允许高达 +9.5 个百分点的差距；
这是更弱的陈述，而且它们从未在 16384 被测试过。

**因此曲线顶端的效率对比仍未定论。** 直接比较：`tree_k256`(16384) vs `quest_B8192` 是 10/5，
p = 0.30，CI [−0.017, +0.090] —— **不显著**。
即 TSA 用两倍 budget 并不比 Quest 用一半 budget 更好。
要主张 TSA 在这个区间更省 budget，必须把 Quest 和 BlockSparse 也跑到 16384。
**尚未跑，因此 8192 以上的任何排序都不被支持。**

### 7.1.2 站得住的结论

- **在紧 budget（4096）下，TSA 显著优于两个 baseline。** 扎实。
- **chunking 而非 scoring 才是关键变量**（quest vs block：4096 时 p = 0.359，8192 时 p = 1.000）。
  扎实，且这正是 TSA 的核心主张。
- **在这个工作负载上 sparse attention 可以做到无损** —— 在 16384，
  即约 29.5k token 上下文的 55%，约 2.7 倍压缩。这是真实但幅度有限的压缩比。
- **一个尚未核实的 caveat：** budget 按构造是 `page_size × top_k`，但 TSA 的 chunk 是*变长*的
  （16–256 token），而它的计账单位是 64。TSA 实际 attend 的 token 数是否像 fixed chunking 一样
  紧贴名义 budget，**未经验证**。若 TSA 在相同名义 budget 下系统性地读取更少的真实 token，
  则它在 4096 的优势被低估、在 8192 的劣势被高估。这是一小时的 instrumentation 工作，
  应当在这些数字发表前完成。

### 7.2 有多少"失败"其实是指标本身

Full attention 的 agree 是 51.1%。在它的 190 个 index step 中，
**57 个（30%）选择了有效但不同的元素**。其中一些是真错误，很多是替代的有效路径。

因此 `agree` 在这份数据上的实际天花板大约在 51–81% 之间（agree ≤ 天花板 ≤ valid），而不是 100%。
**评判 sparse 配置应当以约 51% 为基准，而不是 100%** —— 这让 `block_B8192` 的 47.9%
看起来与孤立阅读时截然不同。

### 7.3 这些 accuracy 结果**没有**证明什么

以上全部是 **offline replay agreement**，不是 task success。
每一步都回放参考上下文，误差不会像真实 agent loop 那样累积。
本项目早期恰好观察到这个落差：replay 探针估计 46.4% 的 index 无效，
而真实 agentic loop 产生了 63.3%。
**这六个配置的 online end-to-end success 从未被测量，而那才是最终重要的指标。**

---

## 8. Speed 结果

### 8.1 最稳健的发现：上下文长度不改变 decode latency

Full attention，batch 1，每 decode step 毫秒数，跨三个独立编写的 harness：

| Harness | 25k 字符 | 50k | 75k | 101k | 118k |
|---|---|---|---|---|---|
| `speed_bench.py` | 138.6 | 155.6 | 152.8 | 139.5 | — |
| `bench2.py` | 150.5 | — | — | 140.9 | — |
| `bench5.py`（最终） | **134.9** | — | — | — | **134.5** |

跨 4.7 倍 KV 范围**没有趋势**，而带宽预测是 +10.2 ms。

### 8.2 153.5 ms 到底花在哪

在真实 tensor 形状下直接实测，batch 1，约 29.5k token：

| 组成 | ms/step | 占比 | 获得方式 |
|---|---|---|---|
| **MoE experts（`grouped_mm`，静默默认）** | **77.1** | **50%** | 实测，1.607 ms/layer × 48 |
| q/k/v/o projection + norm + router + top-k | 13.1 | 9% | 实测，0.273 ms/layer × 48 |
| `lm_head` GEMV (2048 × 151936) | 3.4 | 2% | 实测 |
| **paged attention 读完整 29.5k KV** | **≥13.0** | **≥8.5%** | roofline，60.3 MB/layer ÷ 222.7 GB/s |
| sparse page selection（4 个 kernel + top-k，×48） | 1–4 | 1–3% | 由 launch 数推断 |
| 残余：rope、residual add、Python decode loop、每 token `argmax().item()` + `tokenizer.decode` | ~40–45 | ~28% | 相减得出 |

**Attention 至多是第四大项。** 测量给出的 95% 上界是 22 ms（15%），roofline 下界是 13 ms（8.5%）。
两者都同意它是少数项。

### 8.3 MoE kernel 才是真瓶颈

`Qwen3VLMoeTextExperts` 带有 `@use_experts_implementation`，在未指定时
`modeling_utils.py:1971` 解析为 `"grouped_mm"`。在 batch 1、真实形状下实测
（128 experts，gate_up [128,1536,2048]，down [128,2048,768]）：

| 后端 | ms/layer | ms/step | 相对默认 |
|---|---|---|---|
| `grouped_mm` | 1.607 | **77.1** | —（实际运行的） |
| **`batched_mm`** | **0.925** | **44.4** | **快 1.74 倍** |
| `eager` | 7.115 | 341.5 | 慢 4.4 倍 |
| roofline（8 个激活 expert，75.5 MB） | 0.339 | 16.3 | — |

以 8 个激活行扫描 expert 总数，得到 E = 8/16/32/64/128 时分别为
1.887 / 1.672 / 1.580 / 1.634 / 1.209 ms/layer —— **开销与 E 无关**，
这排除了"它把 128 个 expert 全部物化"的假设（那需要 58 GB/step = 260 ms，比整个实测 step 还长）。
它读取约 8 个 expert 的字节量，却花了该流量成本的 4.7 倍：
一个被交了 128 个 group（其中 120 个是空的）、M=1 的 CUTLASS grouped GEMM。
这是 occupancy 问题，不是带宽问题。

**Caveat：** 在 batch 8 顺序反转（`grouped_mm` 5.364 vs `batched_mm` 6.962 ms/layer）。
后端应当**按 batch size 分别固定**，而不是全局翻转。

### 8.4 CUDA graph：`batched_mm` 可捕获，但真实 prompt 下 OOM

§5.5 的判断是错的。真实答案：

- `grouped_mm` **不能**被 graph 捕获：`RuntimeError: Cannot copy between CPU and CUDA tensors
  during CUDA graph capture`。**这就是整个项目关闭 CUDA graph 的根因** —— 与 §8.3 同源，
  一个 flag 同时修两个问题。
- `batched_mm` **可以**被捕获。服务器打印了 `[CudaGraph] Capture complete`、`Captured for bs=1`，
  并正确回答了一个短请求。
- **但在真实的 5,655-token 浏览器 prompt 下容器被 OOM 杀掉**
  （`OOMKilled=true`，退出码 0，零 Python 错误）。
  graph memory pool + 30B 权重 + KV 超出了 119 GB unified memory。
- 降到 `--max-decode-tokens 512 --max-batch-size 1` 重试仍然崩溃。

在一个合成的 8-layer stack 上，graph 捕获价值 **6%**（9.328 → 8.747 ms）。
真实模型（含 rope、per-head norm、attention kernel、每层 4 次 selector launch）大概是 10–15%。
无论如何都远小于 §8.3。

**CUDA graph 会改变 accuracy 吗？** 不应该 —— graph 以相同顺序重放相同的 kernel，
输出应当逐位相同。这项验证已排期，但被上述 OOM 阻塞。

### 8.5 各配置速度对比 —— 最终结果

v4 网格的排名已撤回（§5.3）。重跑使用 `bench5`（5 个有杠杆的点 × 3 次重复，报告标准误），
仅 batch 1，全部 10 个配置在同一台机器上。**零 length violation，R² 全部 ≥ 0.99。**

**排除 batch > 1**，因为 `serve.py:178-190` 每请求每层执行 4 次 GPU→CPU 同步 ——
每步 192·B 次强制停顿，B=8 时 1,536 次 —— 是实质不同的代码路径。

| 配置 | Budget | 25,036 字符 | ±SE | 118,360 字符 | ±SE |
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
| | | *跨度 14.6* | | *跨度 13.6* | |
| | | *显著性门槛 ≈15* | | *≈9* | |

**三个结论，现在建立在经过验证的估计量上：**

1. **上下文不变性已成定论。** Full attention：**25k 字符 134.9 ms，118k 字符 134.5 ms** ——
   KV cache 增大 4.7 倍，耗时变化 0.4 ms。这里不再有任何解释空间。
2. **Sparsity 换不到任何东西。** `tree_B2048` 做了 12 倍稀疏化，落在 134.1 ms，
   而 full attention 是 134.5 ms。全部 10 个配置的跨度（13.6–14.6 ms）
   等于或低于由实测标准误推出的显著性门槛（≈9–15 ms）。
   **§9 在数据存在之前就预测跨度会小于 15 ms；实测是 13.6 和 14.6。**
3. **残余排序仍然反物理** —— 读取 KV *最多*的 `tree_B8192` 在两个上下文下都是最慢的 cell ——
   这是噪声的样子，不是机制。**不要把它当成机制来报告。**

### 8.6 MoE 后端的收益未能延续到 end-to-end（且第一次对照是无效的）

§8.3 的 microbenchmark 预测 `grouped_mm` → `batched_mm` 每步省 33 ms。做了一次对照实验 ——
结果**在构造上就是空的**：那台机器上 `qwen3vl_inference.py` 的环境变量补丁从未被应用，
所以 `TSA_EXPERTS_IMPL=batched_mm` 是惰性的，*两臂都跑了* `grouped_mm`。

这个意外并非毫无价值。两个名义上完全相同的配置在 25k 下相差 **−2.0 ms**、
118k 下相差 **+4.9 ms** —— 这是对噪声地板的一次独立读数，
与由标准误推出的 ≈9–15 ms 门槛相互印证。

**因此 33 ms 这个主张是"未被端到端检验"，而不是"已被推翻"。**
修正后的对照（补丁已应用，并在服务器日志中**核实** backend 行而非假设）正在运行。
在它返回之前，§10-D1 的"一个环境变量省 21%"只是 microbenchmark 结果，引用时必须如此标注。
孤立的 microbenchmark 经常无法在集成后存活 —— 重叠执行、内存状态、cache 效应都不同 ——
而 end-to-end 才是要的那个数。

---

## 9. 为什么 sparse attention 在这里赢不了 —— 算术推导

Amdahl 定律：`speedup = T / (T − A + A·b)`，其中 `A` 是 attention 时间，`b = budget/context`。

| 情形 | T (ms) | attention 占比 | *完美*稀疏的天花板 |
|---|---|---|---|
| 实测，29.5k tokens | 153.5 | 8.5% | **1.086x** |
| 实测，6k tokens | 155.5 | 1.8% | **1.012x** |
| + `batched_mm` | 95–120 | 11–14% | 1.12–1.16x |
| + CUDA graph | 75–105 | 12–17% | 1.14–1.20x |
| + MoE GEMV 达到 roofline | 45–60 | 22–29% | 1.28–1.40x |
| 理想机器（100% 带宽利用率） | **40.3** | **32.3%** | **1.43x** |

**即使在完美的机器上，这个模型在这个上下文长度下的天花板也只有 1.43x。**
在当前引擎上是 1.09x —— 低于我们一夜之内能搭出的任何 benchmark 的噪声地板。

### 9.1 与 Quest 已发表数字的对照

Quest（Tang, Zhao, Zhu, Xiao, Kasikci, Han — ICML 2024）报告
**attention kernel 单独 7.03x**（Fig. 12），**end-to-end FP16 1.74x**
（Fig. 13；Llama-2-7B，32k 上下文，budget 2048）。
*（arXiv/PMLR 摘要把这两个数字写反了；正文是对的。）*
我们的对比是 end-to-end 对 end-to-end，不存在类别错误 —— 差距来自算术：

| | Llama-2-7B @ 32k | Qwen3-VL-30B-A3B @ 29.5k |
|---|---|---|
| 每 token KV | 512 KiB（MHA，32 KV heads） | 96 KiB（GQA 8:1，4 KV heads）—— **少 5.33 倍** |
| 每步读取权重 | 13.48 GB | 6.083 GB（MoE，3B 激活）—— **少 2.22 倍** |
| KV 总流量 | 16.8 GB | 2.90 GB |
| **attention 占理想 step 的比例** | **55.4%** | **32.3%** |
| budget 2048 的 Amdahl 天花板 | 2.16x（**实测 1.74x**） | **1.43x** |
| *本引擎上*的天花板 | — | **1.09x** |

两个乘性惩罚：

- **架构带来的 0.66x** —— GQA 8:1 缩小分子，MoE 缩小分母的幅度小于它缩小分子的幅度。
  **不可修复。** 这就是这个模型本身。
- **实现带来的 0.76x** —— 那约 113 ms 与上下文无关的开销。**可修复**，§8.3 是其中大头。

**没有任何异常发生。** 一个在 dense MHA 模型 32k 上给出 1.74x 的方法，
在 GQA-8:1 的 MoE 模型 29.5k 上给出约 1.0x，是可预测的结果，不是 bug。

### 9.2 Sparse attention 在这个模型上什么时候才**会**有收益

- **上下文长度。** 在理想机器上，attention 在约 **62,000 tokens**（约 248k 字符）时达到 step 的 50%。
  在当前引擎上则需要约 356,000 tokens —— 超出模型窗口。
  本数据集中 118k 字符的 prompt 约 29.5k tokens：**还不到"这件事开始变得有意思"的一半。**
- **Batch size —— 这是最强的杠杆，本报告先前把它写反了。** 见 §9.3。
- **模型选择。** 同样的方法在 dense MHA 模型上会好看约 2.4 倍。

### 9.3 Batch size：本报告的一处更正

本报告先前的版本称 "Batch 不是杠杆，独立请求会同时放大 KV 和 expert 流量"。**这是错的。**
两类流量对 B 的依赖并不相同：

- **KV 流量：精确线性于 B。** 每个请求拥有自己的 KV cache，没有任何共享。
- **权重流量：次线性，且封顶。** 每个 token 激活 8 个 expert；batch 内不同 token 可能激活不同
  expert，因此 expert 流量随 B 增长 —— 但在 128 个 expert 处**饱和**。
  而 attention projection 与 `lm_head` 对 B 是**常数**（全 batch 共享）。

分子线性增长、分母次线性且封顶 → **attention 占比必然随 batch 上升。**

上下文 29,500 tokens，budget 2048：

| B | 触及的 distinct experts | 权重 GB | KV GB | **attention 占比** | Amdahl 天花板 |
|---|---|---|---|---|---|
| 1 | 8 | 6.06 | 2.90 | **32.4%** | 1.43x |
| 2 | 15 | 9.23 | 5.80 | 38.6% | 1.56x |
| 4 | 28 | 15.12 | 11.60 | 43.4% | 1.68x |
| **8** | 50 | 25.08 | 23.20 | **48.0%** | **1.81x** |
| 16 | 80 | 38.67 | 46.40 | 54.5% | 2.03x |
| 32 | 110 | 52.26 | 92.80 | **64.0%** | 2.47x |
| 64 | 128（饱和） | 60.42 | 185.60 | 75.4% | 3.36x |

**在 B = 32，attention 占比达到 64%，超过 Quest 论文中 Llama-2-7B 的 55.4%**（§9.1）。
换言之：这个模型并非天生不适合 sparse attention —— **是实验被跑在了 B = 1 上。**
§9 的"1.43x 天花板"是一个 **batch-1 的结论**，先前未加限定地陈述，是错误的。

**两个必须标注的 caveat：**

1. **"触及的 distinct experts" 是假设的重叠曲线，不是实测。** 真实值取决于 router 在这批浏览器
   prompt 上的行为：若不同请求的 token 高度集中在同一批 expert，权重流量增长更慢、
   attention 占比上升**更快**；若分散则相反。这可以直接测量（统计 router top-k 输出在 batch 内的
   distinct expert 数），成本很低，做完才能把上表从"推演"升级为"预测"。
2. **B > 1 的绝对延迟被 D3 抬高了。** `serve.py:178-190` 在 batched path 每请求每层执行 4 次
   GPU→CPU 同步（每步 192·B 次）。但这些同步位于 **selector** 内，`full` 配置同样要付，
   因此**固定 B 下的 full-vs-sparse 对比仍然有效**，只是绝对值偏高。
   §8.5 以此为由完全排除 B > 1 是**过度保守**的。

**正在运行：** 用修复后的 bench5 估计量在 118k 上下文下扫 batch 1/2/4/8，
测 `full`、`tree_B2048`、`quest_B2048`。它同时检验两件事 ——
(a) full attention 的 ms/step 是否随 B 显著上升（B = 1 时它对上下文长度完全不敏感）；
(b) 12 倍稀疏是否在 B = 8 开始兑现，不再淹没于噪声。

---

## 10. 发现的缺陷，按严重性排序

### 真正的 bug

**D1 — `experts_implementation` 从未被固定；静默默认既是最慢的、也是不可 graph 捕获的那个。**
`models_qwen3vl_inference.py:137-140` 只从 `TSA_EXPERTS_IMPL` 读取，而没有任何地方设置该变量。
**Microbenchmark 影响：每步 −33 ms（−21%）** —— 是 29k 下整个 attention 项的 2.5 倍、6k 下的 12 倍。
它同时阻断 CUDA graph（D2）。
**Caveat（§8.6）：这尚未在 end-to-end 层面被确认。** 第一次确认尝试是无效的，
且孤立 microbenchmark 常常无法在集成后存活。在修正后的对照返回前，
把 −21% 当作上界。后端应按 batch size 分别固定，而非全局翻转。

**D2 — CUDA graph 的判定是 harness 假象。** 单次读数即判死、无去抖，加上重启后 sleep 过短。
见 §5.5、§8.4。

**D3 — batched path 每请求每层 4 次强制 GPU→CPU 同步。**
`serve.py:178,179,184,190` —— 其中 `int(pi_indptr[-1].item())` 是对 selection kernel 刚写入的值的
硬停顿。每步 192·B 次；B=8 时 1,536 次。batch-1 路径一次也没有。
**所有 batch>1 的测量都是另一条代码路径。** 修法：让 `page_indptr`/`last_page_len` 全程留在 device 上。

**D4 — `page_bitset` 静默截断。** `csrc_ts_tree_sparse.cu:431` 声明
`__shared__ uint32_t page_bitset[128]`（4,096 pages）并以 `min(..., 4095)` 钳位。
在 `page_size=16` 下这限制在 65,536 tokens；超过后 page 会被**静默丢弃**而不是报错。
29k 下未触发。`bitset_cap` 在 `:429` 被计算却从未使用 —— 这个保护显然本应存在，后来丢失了。

### 按设计工作，但代价高

**D5 — page selection 在单个 SM 上运行，且有单线程尾巴。**
`csrc:896` 以 `grid(1), block(256)` 启动 `select_pages_fused_kernel`；
`:483-493` 的收集循环是 `if (tid == 0)` 遍历 `max_page`。
在 page 16 / 29k 下这是每步 1,842 次串行迭代 × 48 层 = **88,416 次依赖迭代**，
而 page 64 只有 461 次。只有约 0.5–1 ms —— **但它系统性地惩罚恰好是 Quest/BlockSparse 这两臂，
且与这两臂存在的目的（`--scoring-method` 对比）完全混淆。**
它的偏向对 baseline 不利，即对 TSA 有利，**在发表任何 TSA 与 baseline 之间的速度对比前必须修复**。

**D6 — benchmark 设计无法分辨它要测量的东西。** 见 §5.3。

**D7 — 每请求热路径中的 `torch.cuda.empty_cache()`。**
`serve.py:493, 516, 651, 720, 794, 959`。一次 device 同步加上 `cudaFree` 掉所有缓存块，每请求两次。
它落在 intercept 里因此不偏置 `ms_per_step`，但在生产中每请求浪费约 0.1–0.5 s。

---

## 11. 结论

### 关于研究问题

1. **TSA 的核心论点在它设计针对的区间内成立。** 在紧的 4096 budget 下，
   结构对齐的变长 chunking 显著优于 fixed chunking（p = 0.015 / 0.020），
   而两个 baseline 之间的 scoring function 差异不显著（p = 0.359）。
   **chunking 才是关键变量。**

2. **该优势在更宽松的 budget 下不复存在，且 TSA 是 8192 下唯一仍显著落后 full attention 的方法。**
   但在 16384，TSA 精确追平 full attention（97/190，CI ±3.3 个百分点），
   而 Quest/BlockSparse 从未在该 budget 被测试过。**曲线顶端的排序尚无定论。**

3. **Sparsity 导致 index 幻觉。** Dense：0/190。Sparse：1–10%。归因干净。

4. **表面失败中有相当一部分是指标造成的。** Full attention 自身也只有 51.1% agree，
   其中 30% 的 step 是"有效但不同"的选择。评判 sparse 配置应以约 51% 为基准，而非 100%。

### 关于速度问题

5. **Sparse attention 在这个模型、这个上下文长度下不可能有收益，任何实现层面的修复都改变不了。**
   完美稀疏的天花板现在是 1.09x，理想机器上是 1.43x。
   这是 GQA 8:1（KV 少 5.33 倍）加 MoE（权重分母小 2.22 倍）的可预测后果，
   并与 Quest 自己在 dense MHA 模型上发表的 1.74x 一致 —— 那里 attention 占 step 的 55% 而非 32%。

6. **瓶颈是一个被静默默认选中的 MoE kernel，不是 attention。**
   153.5 ms 中的 77.1 ms，比自身 roofline 慢 4.7 倍，microbenchmark 显示一个环境变量值约 33 ms。
   **一个 flag 的收益超过整个算法** —— 但这个数字仍待 end-to-end 确认（§8.6）。

### 建议

**立即执行**
- 按 batch size 固定 `experts_implementation`（D1）。先完成 §8.6 的 end-to-end 确认。
- 在信任任何 batch>1 数字之前修复 D3 的同步链。
- 在发表任何 TSA-vs-baseline *速度*对比之前修复 D5 —— 它目前偏向 TSA。
- 加入真正的 dense 臂；把 `full` 改名为 `topk=all`（§5.6）。

**关于研究主张**
- **停止主张 latency。** 把 TSA 定位在**单位 budget 的 accuracy** 上。
  这个主张在 4096 有支持，且是诚实的。
- **补齐曲线顶端。** 把 Quest 和 BlockSparse 跑到 16384，否则无法与 TSA 的 16384 结果比较。
- **核实 realised budget。** TSA 的 chunk 是变长的（16–256 token）却按 64 计账；
  它实际 attend 的 token 数是否紧贴名义 budget 从未验证。这很便宜，且可能直接改写 8192 那个结论。
- **测量 online end-to-end success。** 这里所有的 accuracy 数字都是 offline replay agreement；
  真实 agent loop 会累积误差，此前观察到 46% → 63% 的退化。
- **若目标是展示方法本身，就换实验设置：** dense MHA 模型，或超过 60k token 的上下文。
  在 Qwen3-VL-30B-A3B 的 29.5k token 上，这个实验被放在了该方法结构上不可能获胜的区间里。

---

## 附录 —— 复现

| 产物 | 路径 |
|---|---|
| 逐步 accuracy 结果 | `simulator/runs/sparse3way-20260721/result_h1_*.jsonl` |
| Full attention 基线 | `simulator/runs/sparse3way-20260721/result_h1_dense.jsonl` |
| 最终速度网格 | `simulator/runs/sparse3way-20260721/speedbench/bench5_all.jsonl` |
| 速度网格（v4，已作废） | `simulator/runs/sparse3way-20260721/speedbench/bench4_all.jsonl` |
| 评估数据集 | `offline_half1.jsonl`（50 tasks / 332 steps）、`half1_task_ids.json` |
| Benchmark harness | `bench5.py` |
| Server | `/workspace/TreeSparseAttention/serve.py` |

服务器启动模式（各配置只有 §2.3 所示的参数不同）：

```bash
TSA_EXPERTS_IMPL=batched_mm python3 serve.py \
  --model-path /models/Qwen3-VL-30B-A3B-Instruct --host 0.0.0.0 --port 10000 \
  --max-decode-tokens 4096 --max-batch-size 8 --batch-collect-ms 150 \
  --served-model-name tree-sparse \
  --page-size 64 --top-k 64 --tree-parse-mode webarena --scoring-method envelope
```

**撰写时仍在运行：** 修正后的 MoE 后端对照（§8.6）。
Quest/BlockSparse 在 16384 的 accuracy 运行（§7.1.1）尚未完成 —— 首次尝试因机器故障失败，需要重跑。
