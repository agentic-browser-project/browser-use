# RoPE、Quest 包络与变长块：原始文献审计

核查日期：2026-09-15。范围：本地 `2026-09-11_anchor_exponent_study.zh.md` §2.3、§2.4、§2.7 的相关工作与机制主张；只做核查，不改论文。临时下载、提取和检索记录在 `/tmp/rope-quest/`。采用 Agent Reach / Exa 检索、Jina 网页阅读及官方网页搜索；结论只引用原始论文或作者材料。页码默认指所链接 PDF 的物理页；ACL 另列印刷页码。版本日期与首次公开日期分开列出。

## 结论

不能把“RoPE 会影响稀疏注意力”“RoPE 使 key 集合在坐标空间更分散”“Quest 类统计在大 page 上变差”或“缩小包络宽度来减少重要性高估”作为本工作的宽泛新发现。它们都有明确、甚至很接近的方法先例。

**本轮未找到的精确组合**是：在同一个 query 的候选集合中混用不同 token 长度的连续块，对 Quest 的 AABB 分数拆出宽度项，并用显式 `L^(-α)` 仅校正这一项；再把该校正与每 token attention mass、频带分解、在线任务效果联结。这个“未找到”仅限下表所列论文和本轮检索，不等于证明无人做过，也不能据此断言首创。

最需要补入比较的是 **ArkVale 与 LServe**，而不只是 AB-Sparse：前者已经修正包络高估，后者已经诊断 Quest 的大 page 统计问题。RoPE 机制方面应至少正面比较 **Loki、SALS、A²ATS、FASA、TriAttention**。

## 文献矩阵

“直接”表示与某个被审计主张直接重合，不表示整套算法相同；“相邻”表示研究相同机制或设计问题，但指标、目标或实现不同。

| 工作及时间 | 可定位的原始证据 | 与本报告重合之处 | 仍然不同的部分 |
|---|---|---|---|
| **Quest**；首次 2024-06-16；本地核查 v2，2024-08-26。[论文 v2](https://arxiv.org/pdf/2406.10774v2) | §3.4，p4：逐维 min/max 与 query 符号给出 page 内最大点积的上界。§4.3.1，p7；Fig.8(a)，p8：criticality estimation 的 **page size 4、8、16、32 延迟比较**。 | 直接：AABB / support-function 评分的来源。报告“没有 page size 消融”不准确。 | 查到的是延迟随 page size 变化；未在此版本找到按 page size 的准确率、bound-gap 或 RoPE 频带消融，更未找到混合块长的宽度归一化。 |
| **ArkVale**；NeurIPS 2024。[会议论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/cd4b49379efac6e84186a3ffce108c37-Paper-Conference.pdf) | §5.2，p7，Eq.(2)、(4)：sphere/cuboid 半径分别取 max、center、mean；明确只有 max radius 保证包含所有 key，另两种可避免重要性高估。§6.2，p8，Fig.7：比较六种体积摘要与 centroid，cuboid-mean 的 page top-k recall 最好。 | **直接且很近**：保留 box 中心，减少半径/宽度，放弃严格上界以改进排序，均已出现。不能只描述成“同样按 bound 排序”。 | `r_mean = mean_i abs(k_i-c)` 是逐维平均偏离，**不是**把 `r_max` 乘以同一个 `L^(-α)`。没有看到混合块长 bias 校正、增长指数或 RoPE 归因。 |
| **LServe**；首次 2025-02-20；核查 v2，2025-04-21。[论文 v2](https://arxiv.org/pdf/2502.14866v2) | §3.5.1–3.5.2，pp5–6；Fig.6，p6：Quest 随 page 从 16 增至 32/64 准确率下降。作者把原因归于 page 统计在 token 太多时趋同、失去代表性。Eq.(2)，p6：小逻辑 page 的 min/max 评分；对物理 page 内逻辑 page 分数取 max。 | **直接**：大 page 的 min/max 统计影响选择质量已被明确识别并修复；比只谈 per-head granularity 的 AB-Sparse 更接近报告问题。 | 修复方式是细分统计粒度与层级取 max；不是针对混合长度候选块的宽度缩放，也未把问题定量归因为 RoPE。 |
| **AB-Sparse**；v1 2026-05-12。[论文 v1](https://arxiv.org/pdf/2605.12110v1) | §2.2，p3，脚注1：mean pooling 与 max-min pooling 是正交的 block representation 选择。§2.3，pp3–4：固定 token budget 比较 16/32/64 的 head recall。§3.4，pp6–7：每 head 的 `K_h=ceil(T/B_h)`。 | 直接：块大小影响选择、不同 head 敏感性不同。 | 主要是 **head 之间**不同 block size；一个 head 内仍按该 head 的固定 block size 划分。因此不等同于同一排序列表混合 6–256 token 块的成本差异，也未给出 `L^(-α)`。 |
| **Loki**；v1 2024-06-04；另核查 arXiv v2 HTML。[作者保存的 v1 PDF](https://siddharth9820.github.io/publications/paper_pdfs/loki.pdf)、[v2 HTML](https://arxiv.org/html/2406.02542v2) | v1 §3.1–3.2，pp3–4，Fig.2：pre/post rotary PCA，多数模型 post-rotary intrinsic rank 更高。§5.2，p7，Fig.3：pre/post 校准出的 PCA 基在若干模型上准确率差异很大。v2 §6.1 对可能的 position-dependent calibration 原因作讨论。 | **直接**反证“此前未发现 RoPE 改变稀疏选择的 key 几何/准确率”。 | 研究低维投影而非 AABB 极差。尤其要区分“PCA 基用 pre-RoPE key 校准”和“在线完全移除 RoPE 的 QK 打分”，二者不是同义词。 |
| **SALS**；首次 2025-10-28；核查 NeurIPS 2025 版。[会议论文](https://papers.neurips.cc/paper_files/paper/2025/file/00a0ebcad584c59dbc439c2af8793638-Paper-Conference.pdf) | §3.1，p4，专门讨论 post-RoPE 的 variance amplification；Fig.1(b)，p3。Appendix A，p20，Fig.4：pre/post key covariance spectra 与保留90% variance所需维数比较。§4.2–4.3，pp5–6，Alg.1：在 RoPE-free latent space 选 token，重建后再做 RoPE/精确注意力。 | **直接**：RoPE 分散 key 集合、提高有效秩、改变压缩/筛选准确率，已有实测与专门方法。 | variance / effective rank 不是 query-weighted AABB width；没有建立本报告 `w(L)`、gap、混合块长归一化的同一结论。 |
| **A²ATS**；首次 2025-02-18；核查 v2，2025-06-03。[论文 v2](https://arxiv.org/pdf/2502.12665v2) | §3.1，pp3–4，Fig.1、Observation 2：RoPE 让语义相似但位置不同的 key 表示分离，降低共享 codebook 的跨输入相似性。§4.1，p5，Eq.(11)–(12)：Windowed RoPE 对远端 token 使用固定相对旋转，从而使 key 侧回到 pre-PE 表示。 | **直接**：位置旋转妨碍可复用的 key 摘要/索引，采用位置解耦予以解决。 | vector quantization / codebook；且改变远端位置处理。不是保持原模型 post-RoPE key 的 box 长度归一化。 |
| **SeerAttention-R**；v1 2025-06-10。[论文 v1](https://arxiv.org/pdf/2506.08889v1) | §2.2，p3：“Pooling-based Compression of K”与“Positional Embedding in AttnGate”：pre-RoPE Q/K 输入；K 的 max/min/average pool 拼接再线性投影；在 gate 内重施 RoPE，块的 position 取首 token。称带位置编码比完全去掉位置编码更准确。 | **直接**：块摘要先在 pre-RoPE 空间形成再加入代表位置，已经是实际稀疏 gate 设计；也显示“移除 RoPE 一定更好”不成立。 | 有训练/蒸馏的 gate，不是 Quest 上界；没有推出只按 L 缩放 width。 |
| **FASA**；首次 2026-02-03；核查 v3，2026-02-28，ICLR 2026。[论文 v3](https://arxiv.org/pdf/2602.03152v3) | §3.1–3.3，pp3–4：按 RoPE 二维 frequency chunks 分解、用与全 head top-k 的 contextual agreement 找 dominant chunks。§4.1，p5：用这些频率维度的 QK 分数在线选 token。 | **直接**：RoPE 频带不同作用、频带选择影响稀疏 token 排序，已有完整研究。 | 选取贡献大的频率子集；报告测的是固定频带的极差随连续跨度增长。频带类别和目标统计不同，不能称完全重复。 |
| **TriAttention**；v1 2026-04-06。[论文 v1](https://arxiv.org/pdf/2604.04921v1) | Fig.2，p2：pre-RoPE Q/K 集中，post-RoPE 呈 arc patterns。§2.3，p3：post-RoPE representative queries 的局限。§3.1–3.3，pp3–5：频带中心与三角级数解释距离偏好。§4，pp5–6：位置、norm、concentration 用于 KV 评分。 | **直接**：近似固定的内容向量经位置旋转形成弧线/分散，及其对 KV 重要性估计的影响，已有机制图和算法。 | 没有针对 Quest 的坐标独立极值 gap，也没有用跨度 L 对包络宽度做幂律校正。 |
| **RAP**；首次 2026-02-01；核查 v4，2026-07-30。[论文 v4](https://arxiv.org/pdf/2602.02599v4) | §3，p3：独立 channel pruning 会破坏 RoPE pairs。§4，pp4–5，Eq.(8)：保留整对维度使选择映射与旋转兼容；p7 的 RoPE Pair Selection Spectrum：频率对保留情况。 | 相邻：RoPE-aware pruning 已有直接先例，旋转二元结构必须尊重。 | 主要是权重/缓存 **通道剪枝**，不是 query-aware token/block pruning 或包络分数。避免只凭标题混为同一问题。 |
| **RoBSA**；ACL 2026-07。[会议论文](https://aclanthology.org/2026.acl-long.46.pdf) | §2.2.2–2.2.3，pp3–4（1030–1031）：MLA 的 RoPE 部分得分可作全分数代理。§2.3.2，p5（1032），Alg.1 p6：按层决定用 RoPE proxy 或 full score，按固定 block 聚合并按累计质量选择。 | 直接：利用 RoPE 与 sparse block selection 的关系，不是空白领域。 | 限定 MLA decoupled RoPE；按实际/代理 token scores 聚合，而不是 min/max box。 |
| **Tactic**；v1 2025-02-17；另有 ICLR 2026 版。本行定位 v1。[论文 v1](https://arxiv.org/pdf/2502.12216v1) | §3.6–3.7，p5：相邻 key 不必相似；centroid dot-product 可辅助排序，**但由于非线性，其 attention score 不等于 token 的平均 attention score**。§4.2–4.4，p6：mean centroid，平均 cluster size 32，按分布拟合估计累计质量。 | 相邻且重要：不等大小语义 cluster、per-token 排序与 mass 估计已被区分。报告把 centroid 型概括成天然解决所有长度问题过强。 | 没有 min/max width；distribution fitting 不是显式 `L^(-α)`。 |
| **Squeezed Attention**；首次 2024-11-14；核查 ACL 2025 版。[会议论文](https://aclanthology.org/2025.acl-long.1568.pdf) | §3.1–3.2，pp3–4（32633–32634）。Eq.(1)，p4：`S_i=exp(q C_i^T)/Σ_j N_j exp(q C_j^T)`，`N_j` 为 cluster 大小，用于估计每 token 重要性再阈值选择。 | 相邻：已有对 variable cardinality 的显式处理、per-token importance 区分；不能说以前 cluster 方法完全不需考虑大小。 | `N_j` 进入 softmax 总质量估计，**不是**缩放某个包络宽度，也不证明 mean logit 能精确预测 mass density。 |
| **ClusterKV**；首次 2024-12-04；核查 v2，2025-06-14。[论文 v2](https://arxiv.org/pdf/2412.03213v2) | §III-B、III-C，p3：K-means/cosine 聚类，用 query 与 centroid 估计 cluster 重要性；按重要性排序后沿 cluster size prefix sum 选择，满足 token budget。Fig.4 p3、Fig.5 p4。 | 相邻：语义聚类、不同 cardinality 的候选集和 token-budget admission 已有；报告说均值打分的总体分类基本正确。 | 没有 min/max envelope 或其长度偏差；不能从这一先例推出 box midpoint 无长度依赖。 |
| **RoFormer**；首次 2021-04-20；核查 v5，2023-11-08。[论文 v5](https://arxiv.org/pdf/2104.09864v5) | §3.2，pp4–5，Eq.(13)–(16)：二维位置旋转及相对位置内积；§3.3 pp5–6：正交旋转保范数。 | 基础先例：频率、角跨度与二维旋转均是已知机制，不能把这层数学本身当成新发现。 | 未研究 Quest（时间更早）、块包络 width 或 `α≈0.25`。它也不提供“RoPE 必然扩大任意样本的每一维极差”的定理。 |

## 对报告现有文字的具体判断

1. **“Quest 没有 page size 的消融”应改为更精确的事实判断。** Quest v2 有 page-size latency study；本轮没有发现其 page-size accuracy / bound-gap study。两个说法不可互换。
2. **“ArkVale 同样按 bound 排序”漏掉最相关的设计。** 严格 bound 是 ArkVale 的候选之一；它还明确采用不再包含全部 key 的较小半径，且平均半径表现更好。其 `cuboid-mean` 是本报告 width shrinkage 最应比较的方法之一。
3. **“最接近的是 AB-Sparse”不完整。** 若问题是大块的 min/max 评分恶化，LServe 更直接；若问题是减少 box 重要性高估，ArkVale 更直接。AB-Sparse 的变长主要跨 head，跟同一 head 内混合长度要分清。
4. **“centroid 型天然不随长度增长，因此无需长度归一化”只是一种受条件限制的直觉。** 均值不是极值，确实没有相同的极值增长项；但 pooling 仍可能受内容/位置与长度关联影响。Tactic 专门区分均值 logit 与平均 attention score；Squeezed Attention 显式使用 cluster cardinality。这两篇不支持把均值摘要当作无条件的 mass-density oracle。
5. **“β≈0.25 是 key 几何的属性，不是 web-agent 数据属性”证据强度不足。** 局部数据、多模型的相近拟合值可以支持可迁移性观察；不能仅由频带相关性把内容、位置跨度、head 分布、RoPE base 与模型训练效应拆开。原文表中低频也显著增长、高频并非每一对都在 L=8 转满一周，尤其不宜把简单频带分组解释成受控因果分解。

上述是审计建议，未对报告或论文作任何修改。

## 更稳妥的新颖性边界

不宜使用：

> 首次发现 RoPE 会让稀疏注意力的 key 摘要变差；首次发现大块的 Quest bound 不准确；首次通过缩小 box 宽度改善 page 选择。

在实验与实现均经核验后，可以考虑的有限表述是：

> 我们研究语义/结构分块导致的混合块长候选集，定量分析 Quest 式坐标包络的 query-weighted width 随块长的变化，并评估仅对宽度项做长度校正对 token-budget selection 的影响。现有工作已研究包络高估修正、page 粒度与 RoPE 几何，但在本轮核查的论文中，未发现同样的混合块长宽度校正及其频带/在线联动评估。

即使采用这段，也应把 `α=0.25` 写成当前模型与长度区间中的经验设定，不写成 RoPE 的普适常数；把频带结果写成“与旋转跨度机制一致”，除非补足 no-RoPE/同内容多位置等受控比较。

## 搜索覆盖、未决项与证据限制

- 检索覆盖：Quest min/max loose bound、length normalization、variable blocks、rotated bounding box、RoPE-aware sparse selection、pre-RoPE、frequency selection；并逐一核查委托列出的 ArkVale、AB-Sparse、Tactic、ClusterKV、Squeezed Attention、Loki、LServe 和 RoFormer。
- 本轮未找到直接针对 Quest page 做 **rotated/oriented bounding box** 并证明与 `L^(-α)` 等价的原始方法。这个负检索结果不构成不存在性证明。PCA 投影、pre-RoPE gate、Windowed RoPE 与旋转 box 是不同设计，不应混称。
- **RESA: Bringing Back What Sparse Attention Ignores with Residual Estimation**（ICLR 2026）被原始 OpenReview PDF 的搜索索引命中。索引中的 p15 文本讨论 residual error 与 RoPE 位置偏置的相关性，并明确未建立普遍定量关系。[原始 PDF](https://openreview.net/pdf?id=ktcq26hMCH)。直接取全文和 Jina 均遇到访问验证，本文件不据此作关键推论；其作用是提醒仍有相邻的 RoPE/稀疏误差研究。不要与另一篇 Rectified Sparse Attention（ReSA）混淆。
- 文献存在相近机制不等于本报告的实测数值错误；本次没有重新计算本地 KV dump。文献也不能代替对 frequency-band width、query 聚合和 attention mass 定义的代码核验。
- 已视觉核对 ArkVale p7 的 Eq.(2)/(4) 与 LServe p6 的 Fig.6 / Eq.(2)，以避免 PDF 文本提取误读关键公式；其余证据采用正文/公式文本定位。
- Agent Reach `check-update`：当前 v1.5.0，为最新，无更新提示。
