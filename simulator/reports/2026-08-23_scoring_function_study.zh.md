# Web Agent 稀疏注意力打分函数研究报告:TSA legacy vs Quest vs Centroid

**日期**:2026-08-23(实验窗口 2026-08-14 → 08-23;仍在运行的实验见 §9)
**模型**:Qwen3-VL-30B-A3B-Instruct(48 层,32 query heads,4 KV heads,head_dim=128),text-only
**硬件**:NVIDIA GB10(sm_121,119 GB unified memory),spark00
**workload**:WebVoyager + GAIA 真实 browser-agent 轨迹(context 6k–37k tokens)
**姊妹报告**:`simulator/report.zh.md`(8 月初的 3-way 方法对比;本报告推翻其中多项结论)

术语约定:**end-to-end** = 在 serving 栈里完整 replay 轨迹、看模型输出动作是否与参考一致;**per-step agreement(agree rate)** = offline replay 里,把参考轨迹(omni 模型)在某一步**原封不动的 context** 喂给被测配置,它输出的 index 是否与参考轨迹在该步点击的 index 相同;分母只算 index-action 步。注意三点:① 这是 **teacher-forced** 的逐步模仿指标——context 来自参考轨迹,不是被测模型自己走出来的,错误不会累积;② 参考只是一条可行轨迹,选了另一个同样合理的元素(例如点搜索框旁边的触发链接)也算 disagree,所以 agree 会**低估**真实正确率;③ **agree 不等于 online accuracy**——online 成功率还取决于错误累积、页面加载、超时、多步恢复,且被短 context 步骤稀释(随机选页在 online 仍有 26% 成功)。agree 回答的是"在完全相同的输入下,选择质量相对 dense 损失了多少",这正是比较打分函数所需的受控指标;online 精度需要单独测(此前的 online 数据因 §3 的两个 bug 作废);**per-layer hit rate** = 每层 page 选择把 needle chunk 选进 budget 的比例;**chunk geometry** = chunk 的切分方式(tree 变长 vs fixed16 定长);**budget accounting** = budget 按 chunk 数(top-k)、token 数还是 page 数计;**TSA default path** = TSA 当前默认的 CUDA selector 路径(`TSA_SCORING_IMPL` 未设置时走的代码)。page / chunk / needle / mass 的定义见下表。

### 术语表(先读这个)

**page vs chunk——不是一个东西,TSA 里是两层**:

| 术语 | 文献里的定义 | TSA 里的定义 |
|---|---|---|
| **page** | Quest:沿用 PagedAttention 的内存页,"selects the KV cache pages at the granularity of pages",每页 16 个 KV 对,"chooses the top K pages as critical, where K is an arbitrarily defined hyper-parameter"(budget 以 token 计:32…4096)。Vortex(`Infini-AI-Lab/vortex_torch`):page = "the unit of sparsity",`page_size=16`(2 的幂,默认取 SGLang 的 page_size),top-k 以 page 计("keep the 30 highest-scoring pages");README 里 **block 与 page 同义**(`block_size: Vortex page size`)。 | FlashInfer 的**内存 / attention 单元**:`page_size` 个连续 token(tree 模式 64,fixed16 模式 16),显存里每页存成 `[2, page_size, H_kv, D]`(§3.1)。attention kernel 以 page 为单位读 KV。 |
| **chunk** | 文献里通常指 chunked prefill 的分段,不是选择单元。 | TSA 的**选择(打分)单元**:tree parser 按 DOM/ChatML 结构切出的变长区间 `FlatChunk(start_idx, end_idx)`,16–256 token;`--tree-parse-mode fixed` 时 `make_fixed_chunks(chunk_size=page_size)`,此时 **chunk ≡ page**。 |
| **block** | block-sparse attention(Block-Sparse-Attention / MInference / SpargeAttn)里指 attention 矩阵的固定 tile(一块 query × 一块 key,如 64×64),稀疏模式按 tile 决定——与本报告的 page/chunk 都不是一回事。Vortex 把 block 当 page 的同义词。 | 本报告不用 block 指代单元;"BlockSparse" 仅指 vortex 的那个 centroid+softmax 打分算法。 |

**TSA 的 chunk → page 映射**(`tree_sparse_selector.py:523-562`,`_chunk_page_ranges` / `_fast_pages`):打分和 top-k 都在 **chunk** 上做(`top_k_chunks`);选中的每个 chunk 映射到它覆盖的 page 区间 `[start_idx // page_size, end_idx // page_size]`,所有区间取并集,再并上 always-include 的页(开头 `always_include_first`、最近 `always_include_recent` 个 token、以及 `always_include_from` 之后所有已生成 token 所在的页),得到 attention kernel 真正读取的 page 集合。由此产生两个后果:① tree 模式下一个 10-token 的 chunk 会把它所在的整个 64-token page 拉进来(同页的邻居白送),跨页的 chunk 会拉进 2 页——所以 **"k=64 chunks" 既不等于 64 页,也不等于 4096 token**;§5.3 的 token 记账用的是选中 chunk 的长度之和,实际 KV 读取量是不同 page 数 × page_size,两者都要声明。② fixed16 模式下 chunk ≡ page,三种记账(chunk 数、token 数、page 数)完全一致,这是 fixed16 对比起来"干净"的原因之一。

**对照**:Quest 的 page = Vortex 的 page/block = TSA fixed16 的 chunk = FlashInfer 的 page(当 page_size=16)。TSA tree 模式的变长 chunk 在文献里没有对应物。

**Vortex 源码原文**(`Infini-AI-Lab/vortex_torch`,`vortex_torch/flow/algorithms.py`,2026-08 HEAD):`BlockSparseAttention`(L21-26):"Keep one centroid per page (the mean of its keys) and select the pages whose centroid best aligns with the query";`GQABlockSparseAttention`(L101-133):"Each page keeps a centroid … each head's per-page scores are turned into a softmax distribution over pages",`self.softmax = Softmax(dim=0, scale=0.09)`;`GQAQuestSparseAttention`(L176-222):"maintains per-page **max** and **min** envelopes of keys",`Maximum()  # elementwise max(q*max, q*min)`,`CMax(dim=1)` / `CMin(dim=1)` 为 "page-wise max/min envelope over k";所有 `create_cache(block_size, head_dim)` 的注释都写 "``block_size`` is the per-block token count"——即 Vortex 的 block 就是 page。另有 `LServeSparseAttention`(L264-273):"QUEST envelopes at **sub-block** granularity … a page is ranked by its single best-matching (head, sub-block) pair"——把 envelope 做到比 page 更细,恰好是针对本报告 §5.3 宽度偏置的一种已有应对思路。

**needle 与 mass——两种不同的 ground truth**:

- **needle**(借用 needle-in-a-haystack 的说法)= 参考轨迹在该步点击的那个元素(例如 `[54]<input …>`)的 token 所在的 chunk。它是**行为上**必需的 chunk:模型要输出 `"index": 54`,那一行 DOM 文本必须在它读到的 chunk 里。**needle hit rate** = 某层、某个 decode step 的选择把 needle chunk 选进 budget 的比例。它不是"attention 的 ground truth chunk"——模型并不一定要把最大 attention 放在那里;它是任务层面的 ground truth(来自参考轨迹),是 end-to-end agreement 的机理代理。
- **mass** = attention mass = softmax 权重。full attention 下每个 token 的权重 `softmax(q·k_i/√d)`,一个 chunk 的 mass = 其中 token 权重之和(逐 head 算再求和);**mass recall@budget** = 选中 chunk 覆盖的总权重比例。这是**attention 层面**的 ground truth(稀疏 attention 对 full attention 的逼近程度);**oracle** = 直接按真实 mass 排序 chunk(只有算了 full attention 才知道,不可部署)。
- 两者为什么不一致:mass 被 BOS sink 和最近的 token 主导(它们本来就由 always-include 保证),而 needle 通常只是一行 DOM、mass 份额很小。因此 legacy 的 mass recall 很高(它总把 sink 页排第一)却 needle hit 低;实测决定 end-to-end 的是 needle 列(§5.1)。toy 里(§1.1 (c)):needle = chunk A 的第一个 key(10 分),mass = A 93.2% / C 6.8% / B 0.02%——这个 toy 里两种 ground truth 恰好指向同一个 chunk,真实页面上往往不是。

---

## 0. TL;DR

(对打分问题不熟悉的读者,建议先看 §1.1 的数值 toy example,再回来读结论。)

1. **TSA 的精度损失根本不是打分函数的问题,而是一个 KV buffer 布局 bug(§3)**:serving 路径把 FlashInfer 的扁平 paged-KV buffer 当成 `[tokens, H_kv, D]` 的行布局去索引,**page 0 之后所有 token 的 key 都读错**(一半读到 value 向量,另一半读到错页的 key)。PyTorch fallback 和 CUDA selector 两条路径犯同一个错;fallback 已于 08-15 修复,**CUDA selector(TSA default path)至今未修**,本报告在真实 serving 路径上用测试复现(§3.3)。git 考古(§3.6)显示这个不匹配从仓库第一个 commit(2026-04-08,原作者)就存在,不是本项目引入的;截至 upstream main 最新提交(2026-08-20,`0457e95`)仍未修复,fallback 的修复也只在 spark00 容器里。
2. 后果:**所有基于 TSA default path 的历史数字——6 月 k32 的 0.41% 在线成功率、8 月初 3-way 报告、本报告表中全部 "legacy" 行——都是在垃圾 page 统计量上做的选择**,水平 ≈ 随机选页 + always-include。end-to-end A/B 直接验证:同一套 legacy 打分数学,换到修复后的 fallback 路径,同一批步子上 per-step agreement 从 **13.8% 升到 58.6%**(§3.4)。
3. 修好布局后,**任何 fp32 的忠实打分(Quest / BlockSparse / mixmax)在 4096-token budget 下 end-to-end 与 dense 无差别**(fixed16 chunk:50.5–52.1% vs dense 51.1%)。"sparse attention 掉精度"的旧结论作废。
4. 打分函数之间的差异在 B4096 是二阶效应;一阶变量是 **chunk geometry**:tree 变长 chunk 一致地比 fixed16 低 4 个点(46.8–47.9%),根因是 envelope 打分的 **width bias**(§5.3)。
5. **机理指标不能跨 chunk geometry 搬运**:BlockSparse 在 tree chunk 上 needle hit rate 全场最低(.478),在 fixed16 上全场最高(.764);Quest 则相反。centroid 的病是 **dilution**(随 chunk 变大加重),envelope 的病是 width bias(同样随 chunk 变大加重)。
6. **budget accounting 会翻转 tree 上的结论**:envelope 打分在 tree 上选的 chunk 平均 108 token(legacy 65),同为 top-64 实际多读 1.6× KV;按 token 对齐后,tree 上 ≤B4096 的赢家变成均值类打分。
7. 6 个 case study(§6)给出四种可复现的选对/选错模式:head 平均淹没检索信号、centroid dilution、width bias 埋掉小元素、softmax+head-MAX 在细 page 上放大噪声。
8. 工程落地:① 修 CUDA selector 的布局(一行 view,§3.5);② `mixmax` 已实现并验证,是机理指标最优、与现有 kernel 同构的打分;③ 宽度归一化 α=0.25 是 token-budget 模式的候选默认(§5.5)。

**Stage B/C/D 追加结论(2026-08-22 → 08-23,详见 §6.9、§6.10、§9)**:

9. **新的 tree 方法 = tree_min16(tiny chunk 合并到 ≥16 token)+ mixmax_wn(宽度归一化 α=.25)+ token 预算**。在 B4096 上它是 tree 几何里唯一与 fixed16 基线和 dense 统计上不可区分的配置(half1 190 步:96 vs Block 99 / Quest 98 / dense 97;旧 tree-Quest 89,p=0.02),valid 最高、被打分的单元约为 fixed16 的 1/4。**但不能宣称超过 fixed16**:B4096 是"透明区",dense 自己对 Block 也是 6 赢 8 输,所有 faithful 方法落在同一噪声带(fidelity 76–80%),agree 指标在 n=190 分不出 <5 pt 的差异;dump 复盘显示 5 个"真损失"里 3 个同配置重跑即变对,其余是 needle 在手的决策错误(§6.9)。
10. **紧预算(B2048)下 tree 明确输给 fixed16**(strat20 132 步:48 vs Block 55 = Quest 55,dense 57;fidelity 57% vs 63–66%)。dump 复盘(§6.10)给出两个机制:① needle 在 109–219 token 的 search-input 子树里时几乎从不入选(0.01–0.20;fixed16 页 0.15–0.66);② system prompt 覆盖从 B4096 的 36–62% 塌缩到 7–14%,模型空转(none 40 vs 25–33)。两者在固定预算下互相冲突。
11. **离线代理指标必须在失败富集的真实 q 流上测**:旧 dump 上的 needle-hit 曾预测 tree 在 B2048 反超(.56 vs .44),online 相反;在 23 个新 dump 上,fixed16-Block 的 needle 最高(.537),所有"修大 chunk"的 tree 变体(拆分、子块取 max、trim64、Block 式打分)都 ≤ .515,且 Block 式打分把 mass 从 .80 拉到 .50。问题是 **head 聚合**(head-mean 抹掉少数 head 的检索信号),不是粒度。
12. 目前诚实的定位:**按预算自适应粒度**——B≥4096 用 DOM 树整块选择(追平 fixed16,打分单元 1/4,结构钩子可用),≤2048 用 fixed16(Block 或 Quest)。tree 特有的杠杆是结构信息(system prompt / DOM 区域的预算保障、元素对齐的读取、跨步 chunk 复用),sys-floor online 验证为部分修复(+2,仍差 5);selection-cost 微基准、B3072 交叉点(**切换阈值 ~3k token**:B3072 tree 57 = dense 57 > Block 53;B2048 fixed16 55 > tree ≤50)与时间复用验证(每 8 步重选:55 vs 58,p≈0.63,wall −21%)均已落地。

---

## 1. 先看一个 toy example,再谈研究问题

### 1.1 Toy example:打分在什么东西上打、full vs sparse、各打分方法的差别(带具体数值)

**(0) q、k、v 分别是什么、从哪来。** Transformer 每一层里,每个 token 经过该层时,由它在这层的隐状态 h(由 token 本身和它前面的上下文经前面各层算出)线性投影出三个向量:q = W_q·h(query)、k = W_k·h(key)、v = W_v·h(value);W_q / W_k / W_v 是训练好的固定权重。用户 prompt(系统提示 + 任务 + 整页 DOM 文本,一两万 token)在生成开始前被一次性处理(**prefill**),每个 prompt token 在每一层算出自己的 k 和 v 并存入 **KV cache**,之后不再变化——所以 k、v 是"固定"的,但它们**不是与 prompt 独立,而恰恰就是 prompt 的表示**;它们独立的是后面的 decode。这正是为什么 chunk 的摘要(envelope 的 M / m、centroid)可以在 prefill 时算一次、之后每个 decode step 反复用。q 则属于"当前正要生成的这个 token":每个 **decode step**,当前位置在每一层、每个 head 算出一个新的 q(也由上下文决定,但代表的是"现在在找什么"——模型正要写 `"index": 54` 时,q 编码的就是"我在找搜索框",而 DOM 里 `[54]<input …>` 那几个 token 的 k 会与它有很高的点积)。因为每一步的 q 都是新的,所以每个 decode step、每一层都要重新选 chunk。

**KV cache 存的是什么**:对当前请求里每一个已处理的 token 位置、每一层、每个 KV head,存一对 (k, v);本模型每 token ≈ 98 KB(48 层 × 4 KV head × 128 维 × 2 向量 × bf16),2 万 token 的页面 context ≈ 2 GB。内容 = prompt 经 prefill 算出的全部 k、v + decode 过程中逐个追加的新生成 token 的 k、v;范围是**一个请求**——web agent 每走一步都发一个新 prompt(历史记录作为文本重新发来),server 重新 prefill、建新 cache、生成完即释放。缓存的意义:否则每生成一个 token 都要把前面所有 token 的 k、v 重算一遍;缓存后每个 decode step、每一层只需"读一遍 cache"——而这个读取量正是 sparse attention 要砍的:budget 4096 token 意味着每层只读 4096 个 token 的 (k, v) 而非全部。cache 在显存里按 page 组织(每页 page_size 个 token,布局 `[2, page_size, H_kv, D]`,先全部 k 再全部 v,即 §3 中被错误索引的那个布局),选 chunk 就是选 page。toy 表 (b) 的 12 个 key 就是 prefill 一个 12-token prompt 后某一层 cache 里的全部 k(v 也在 cache 里,选 chunk 用不到所以未画)。

**v 的去向**:选 chunk 只用 k(及其摘要);选定后 attention kernel 只对选中 chunk 里的 token 做正常 attention——用它们的 k 算精确的 softmax 权重,再用它们的 **v** 加权求和得到输出。未选中 chunk 的 k、v 都不读,省下的正是这部分 KV 读取量。信息通过 v 进入当前 token 的表示:k 决定"读多少",v 决定"读到什么"。

**(a) 一次 decode step 里发生了什么。** LLM 生成第 n 个 token 时,每一层的每个 attention head 得到一个 query 向量 q(维度 D;真实模型 D=128),要和 KV cache 里此前 n−1 个 token 各自的 key 向量 k_i 做点积 q·k_i,softmax 之后按权重把对应的 value v_i 加权求和——这就是 **full attention**,每个 decode step、每一层都要读全部 n−1 个 (k, v),成本正比于 context 长度(web agent 的 context 是 1–3 万 token)。在 web agent 场景里,"该点哪个元素"的信息正是藏在 q 与少数几个 key(目标元素那一行 DOM 文本的 token)的高点积里。

**Sparse attention** 的想法:把 KV cache 按位置切成 chunk(连续若干 token),每个 decode step 只读 budget 内的少数 chunk。于是问题变成:**不读 key,怎么知道该读哪些 chunk?** 答案是 prefill 时给每个 chunk 算一个廉价的**摘要**(envelope 的逐维 max/min,或 centroid),decode 时只拿 q 和摘要打分、排序、取 top-budget,再只对这些 chunk 的 k、v 做 attention(未选中的 chunk 一个字节都不读)。**"打分方法" = 用什么摘要 + 怎么拿 q 和摘要算分。** 选错 = 把真正高点积的 key 所在 chunk 排到了 budget 之外,模型就看不到那个元素。

**(b) Toy 设定。** D=4,12 个 token 切成 3 个 chunk、每个 4 token(真实:D=128,每层数百到数千个 chunk,每个 chunk 16 token(fixed16)或 16–256 token(tree),budget 64 个 chunk 或 4096 token)。当前 query:

`q = [2, −2, 1, −1]`

| chunk | 4 个 key 向量 | q·k(full attention 的 logit) | 现实对应 |
|---|---|---|---|
| **A**(needle chunk) | [2,−2,1,−1] · [0,0,0,0] · [−1,1,0,0] · [0,0,−1,1] | **10**, 0, −4, −2 | 一个强匹配 key + 三个无关 token:含目标元素的 DOM 子树 |
| **B**(大而杂) | [3,3,0,0] · [−3,−3,0,0] · [0,0,3,3] · [0,0,−3,−3] | 0, 0, 0, 0 | 内容分散、没有任何 key 匹配 q:导航栏、链接堆 |
| **C**(小而同质) | [1,−1,1,−1] × 4 | 6, 6, 6, 6 | 四个几乎相同的中等匹配 key:cookie 按钮那类短 chunk |

**(c) Full attention 的答案。** 对 12 个 logit 做 softmax:A 的第一个 key 拿到 **93.2%** 的权重,C 的四个各 1.7%(合计 6.8%),B 合计 0.02%。如果 budget 只够读 1 个 chunk,**正确答案是 A**:读 A 拿到 93% 的 attention mass 和那个决定性的 key;读 C 只拿到 6.8%;读 B 等于什么都没读。报告里两个指标由此而来:**needle hit** = 是否选中了含最强 key 的 chunk;**mass recall** = 选中 chunk 覆盖的 softmax 权重之和。

**(d) 每个 chunk 的摘要**(prefill 时算一次,与 q 无关):

| chunk | M(逐维 max) | m(逐维 min) | c=(M+m)/2 | w=(M−m)/2 | centroid(mean) |
|---|---|---|---|---|---|
| A | [2, 1, 1, 1] | [−1, −2, −1, −1] | [.5, −.5, 0, 0] | [1.5, 1.5, 1, 1] | [.25, −.25, 0, 0] |
| B | [3, 3, 3, 3] | [−3, −3, −3, −3] | [0, 0, 0, 0] | [3, 3, 3, 3] | [0, 0, 0, 0] |
| C | [1, −1, 1, −1] | [1, −1, 1, −1] | [1, −1, 1, −1] | [0, 0, 0, 0] | [1, −1, 1, −1] |

**(e) 三种打分在同一组数字上的结果**(|q| = [2, 2, 1, 1]):

| 打分 | 公式 | A | B | C | budget=1 时选谁 |
|---|---|---|---|---|---|
| **Quest**(elementwise max) | Σ_d max(q_d M_d, q_d m_d) = q·c + \|q\|·w | 2 + 8 = **10** | 0 + 18 = **18** | 6 + 0 = 6 | 有 B 时选 B ✗;没有 B 时选 A ✓ |
| **TSA legacy**(corner-max) | max(q·M, q·m) = q·c + \|q·w\| | max(2, 2) = **2**(q·w = 3−3+1−1 = 0) | max(0, 0) = 0 | 6 | **C ✗** |
| **centroid** | q·mean | **1** | 0 | 6 | **C ✗** |

逐条对应本报告的结论:

1. **Quest 给 A 的 10 正好等于真实最强 key 的 q·k。** elementwise max 是 q·k 在 chunk 包围盒(每维取 [m_d, M_d])上的 **upper bound**;A 的 needle 在每个维度上都恰好是极值,所以 bound 是紧的。这就是 envelope 的价值:回答"这个 chunk 里最好的 key 能有多好"(§5.1、案例 1)。
2. **legacy 只给 A 打 2。** q 的各维符号混合(+,−,+,−),两个角 q·M 和 q·m 里宽度项正负相消(3 − 3 + 1 − 1 = 0),只剩下中心项 q·c = 2——**corner-max 在数学上退化成"用 chunk 的中心打分"**,needle 的信息被丢掉(§1.2 偏差 1;rank 相关 0.87 的来源)。于是它选了 C。
3. **centroid 给 A 打 1。** needle(10)被三个无关 token 平均掉——这就是 **dilution**(案例 3)。也选 C。
4. **Quest 给 B 打 18,高于 A 的 10。** B 里没有任何 key 匹配 q(全是 0),但它的包围盒很大(w = 3),`|q|·w` 项白送 18 分——**width bias**(§5.3、案例 5):upper bound 只有在 chunk 里真有 key 坐在那个乐观角上时才紧,杂乱的大 chunk 包围盒大但角上没人。legacy 和 centroid 对 B 都给 0(它们不看宽度),反而不上当——这就是 token 对齐后 tree 上均值家族反超(§5.3)、以及忠实打分在 tree 上比 fixed16 低 4 个点的原因。
5. **C 在三种打分下都是 6,全部精确。** 四个 key 完全相同,包围盒宽度为 0,centroid 就是 key 本身——**小而同质的 chunk 是 centroid 的主场,envelope 没有任何加成**(案例 5 的 cookie 按钮)。
6. **宽度归一化**(§5.5):若 B 是 256-token 的大 chunk、A 是 16-token,α = 0.25 把 B 的宽度项乘以 (16/256)^0.25 = 0.5 → B = 9 < A = 10,排序恢复;α 再大又会把 A 自己的宽度项(8 分,占其总分 80%)也压掉——这就是"宽度项既是 needle 信号也是尺寸偏置"的两难。
7. **对真实 chunk geometry 的推论**:fixed16 的 chunk 只有 16 个 token,包围盒普遍窄、dilution 轻,所有 chunk 都接近"C 型",三种打分的差异消失(transparent regime,§4.1);tree 的 16–256 token 变长 chunk 同时制造"A 型"(envelope 赢)和"B 型"(envelope 输),两种病一起放大。

**(f) 多个 head 的影响**(偏差 2、3)。真实模型一层有 32 个 query head、共用 4 组 KV(每组 8 个 query head 共用同一套 key),而 page 选择是**所有 head 共用一个结果**,必须把各 head 的分数聚合。Toy:两个 head,head 1 的 q1 = [2, −2, 1, −1](指向 needle 的"检索 head"),head 2 的 q2 = [0, 0, 1, 1](对 needle 无感):

| | A | C | 选谁 |
|---|---|---|---|
| 逐 head 的 Quest 分 | head1 = 10,head2 = 2 | head1 = 6,head2 = 0 | |
| head-MAX(Quest 的做法) | 10 | 6 | A ✓ |
| head-MEAN | 6 | 3 | A ✓(优势减半) |
| legacy:先把两个 q 平均成 q̄ = [1, −1, 1, 0],再 corner-max | max(2, 0) = **2** | 3 | **C ✗** |

先平均 query 再打分,等于让无关 head 稀释检索 head 的投票,之后 corner-max 再丢掉宽度——Coursera 案例里 q̄·c 变成负数就是这个机制。实测中 head-MAX vs head-MEAN 本身只是二阶差异(quest 与 quest_hm 的 rank 相关 0.98),主导偏差是 corner-max;而在 fixed16 的数百个细 page 上,head-MAX 反而引入单 head 假阳性噪声(§5.2)。

**(g) §3 的 bug 在这个 toy 上具体发生了什么。** 设 page_size = 4,12 个 token = 3 页 = 恰好 chunk A / B / C。KV cache 在显存里的真实布局是**一维**的,每页先放 4 行 k、再放 4 行 v(每行一个 token 的向量):

| 扁平 buffer 的行号 | 实际内容 |
|---|---|
| 0–3 | page 0 的 K 块:token 0–3(chunk A)的 k |
| 4–7 | page 0 的 V 块:token 0–3(chunk A)的 v |
| 8–11 | page 1 的 K 块:token 4–7(chunk B)的 k |
| 12–15 | page 1 的 V 块:token 4–7 的 v |
| 16–19 | page 2 的 K 块:token 8–11(chunk C)的 k |
| 20–23 | page 2 的 V 块:token 8–11 的 v |

正确寻址:token t 的 k 在第 `(t // 4)·8 + t % 4` 行。bug 代码(修复前的 fallback、以及至今未修的 CUDA selector)读的是**第 t 行**:

| chunk | 代码以为自己在读 | 实际读到的 |
|---|---|---|
| A(token 0–3) | 行 0–3 | A 的 k ✓ |
| B(token 4–7) | 行 4–7 | **A 的 v** ✗(value 向量,根本不是用来和 q 匹配的) |
| C(token 8–11) | 行 8–11 | **B 的 k** ✗(错了一页) |

给 A 的 v 取几个具体值(v 的内容本来就与"匹配 q"无关,例如 [0,0,0,1]、[0,0,1,0]、[0,1,0,0]、[1,0,0,0],其 M = [1,1,1,1]、m = 0),bug 下表 (d) 的摘要和表 (e) 的打分变成:

| chunk | bug 下摘要实际来自 | Quest | legacy | centroid | 真实应得(表 e) |
|---|---|---|---|---|---|
| A | A 的 k(正确) | 10 | 2 | 1 | 10 / 2 / 1 |
| B | **A 的 v** | 0 + 3 = 3 | 0 | 0 | 18 / 0 / 0 |
| C | **B 的 k** | 0 + 18 = **18** | 0 | 0 | 6 / 6 / 6 |

Quest 选中"C"——那 18 分其实是 B 的 junk 包围盒;真实的 C(6 分)和真实的 B 从未被评估过。之后 attention kernel 去读 page 2 的 k、v——**读本身是对的,错的只是"选了哪一页"**。toy 里 A 恰好在 page 0 所以摘要正确;真实 context 的 page 0 是 system prompt 的前 16–64 个 token,needle 从不在那里,而其余数百到数千个 chunk 的摘要全部来自 v 或错页的 k,排序与真实内容无关——这就是所有 legacy 行 ≈ 随机选页水平、而修好路径后 legacy 数学能与 Quest 打平(§3.4)的原因。§3.3 的单元测试就是这个 toy 放大到 256 token、page_size 16:K 行填 token id、V 行填 −1000,bug 路径选了 chunk 14(读到的是 page 7 的 k)而不是 15(读到的是 V 块的 −1000)。

**(g′) 逐步计算:代码 → 下标 → 读到的向量。** 用 toy 的参数(H_kv=1,D=4,page_size=4,3 页)把涉及的三段代码各算一遍。

*第 1 步,buffer 是怎么摆的*(`models/qwen3vl_inference.py`,`convert_kv_to_interleaved`;serve.py 在 prefill 之后立刻调用它):
```python
key_pages   = key_states.reshape(batch, H, num_pages, page_size, D).permute(0, 2, 3, 1, 4)   # [pages, ps, H, D]
value_pages = value_states.reshape(...).permute(0, 2, 3, 1, 4)
kv_interleaved = torch.stack([key_pages[0], value_pages[0]], dim=1)   # [pages, 2, ps, H, D]:每页先 K 块后 V 块
kv_flat = kv_interleaved.flatten()                                    # 一维
```
每页 `elems_per_page = 2·ps·H·D = 2·4·1·4 = 32` 个数。token t 的 k 的第 d 维在扁平下标 `((t//4)·8 + t%4)·4 + d`,它的 v 在 `((t//4)·8 + 4 + t%4)·4 + d`。例如 token 5(chunk B 的第 2 个 token):k 在下标 36–39,v 在 52–55;token 9(chunk C 的第 2 个):k 在 68–71。

*第 2 步,selector 怎么取*(`csrc/ts_pybind.cu:246`:`pool_size = k_buffer.size(0)`——一维 buffer 时等于元素总数 96,不是行数;`csrc/ts_tree_sparse.cu:81-82`:`k_idx = physical_loc·H·D + head·D + d`,H=1 时就是 `t·4 + d`):

| 要读的 token | 代码算出的下标 | 该下标在真实布局里是什么(用第 1 步的公式反推) |
|---|---|---|
| 5(chunk B) | 20–23 | 行 5 → page 5//8 = 0,半区 (5//4)%2 = 1 = **V**,槽位 1 → **token 1 的 v** |
| 9(chunk C) | 36–39 | 行 9 → page 1,半区 (9//4)%2 = 0 = K,槽位 1 → **token 5 的 k**(错了一页) |
| 2(chunk A) | 8–11 | 行 2 → page 0,半区 0 = K,槽位 2 → token 2 的 k ✓ |

越界检查 `physical_loc < pool_size` 永远成立(token 号 ≤ 11 ≪ 96),所以没有任何报错。

*第 3 步,效果*:chunk B 的 envelope 由 A 的四个 v 算出,chunk C 的 envelope 由 B 的四个 k 算出,即上表 (g) 的 3 / 18;fallback 修复前更糟——一维 tensor 用 token 号做 fancy-index 取到的是单个标量,envelope 整个退化成一个数。

*第 4 步,在真实 serving 路径上复核*(`scoring_case_study/test_cuda_layout_realpath.py`,§3.3):H_kv=4、D=128、page_size=16、prefill 512 token、真实的 `convert_kv_to_interleaved` + `build_shared_prefix_kv`,把 token 100 的 key 设成 needle——测得 token 100 的 key 落在扁平第 **196** 行(= (100//16)·32 + 100%16,与第 1 步公式严格一致),selector 为 token 100 读的第 100 行里装的是 **token 52 的 key**,于是 CUDA selector 把 needle 记到了 **chunk 12**(= 196//16,即"行号被当成 token 号"时它所属的 chunk),而修复后的 fallback 正确选中 chunk 6。

### 1.2 研究问题

TSA、Quest、BlockSparse 都把 KV cache 切成 chunk/page,对每个 chunk 相对当前 query 打分,只对 top-budget 的 chunk 做 attention。打分设计:

| 家族 | chunk 摘要 | 单 head 打分 | 含义 |
|---|---|---|---|
| **envelope**(Quest、TSA) | 逐维 max/min:M, m | `Σ_d max(q_d·M_d, q_d·m_d)` | q·k 在 chunk 包围盒上的精确 **upper bound** |
| **centroid**(BlockSparse) | mean-key 质心 | `q·centroid`(+softmax) | chunk "平均像什么" |

记 `c=(M+m)/2`(center)、`w=(M−m)/2`(half-width),Quest 单 head 分数 = **`q·c + |q|·w`**。`|q|·w` 是 envelope 的全部价值:chunk 里藏着一个强匹配 key 时绝不低估它。

**TSA legacy 打分与 Quest 有三处偏差**(`csrc/ts_pybind.cu:393-415`、`csrc/ts_tree_sparse.cu:386`、`python/tree_sparse_selector.py:750-756`):

1. **corner-max 代替 elementwise max**:`max(q̄·M, q̄·m)` = `q̄·c + |q̄·w|`。q 各维符号混合时 `|Σ_d q_d w_d| ~ √D`,而 `Σ_d |q_d| w_d ~ D`,乐观项缩水到约 1/√128 ≈ 9%——legacy 退化为 midrange-centroid(实测 rank 相关 0.87)。
2. **query head 组平均**:每组 8 个 query head 先平均成 q̄(Quest 逐 head 打分)。
3. **head aggregation 取 MEAN**:4 个 KV head 的分数取平均(Quest 对 32 head 取 MAX)。

另有 fp8 量化(实测无影响,`legacy_fp32 ≈ legacy`,Δ≤.002)。

研究问题:这些偏差在真实 web-agent workload 上值多少?什么时候选对/选错 chunk?centroid 家族(含 BlockSparse 原版)处于什么位置?结论如何随 chunk geometry 与 budget 变化?——**而回答这些问题的过程中,发现真正决定性的不是打分数学,而是 §3 的 bug。**

---

## 2. 实验体系

**两类互补测量**:

- **end-to-end offline replay**(`traj_eval.py`):half1 = 50 个 omni-success 任务的完整轨迹,332 步(其中 190 个 index-action 步)。每步把真实 context 喂给被测配置的 `serve.py`,temperature 0 + xgrammar 约束,判定输出动作的 agree(index 与参考一致)/ valid(index 在页面有效集内)/ none(非 index 动作)。同配置重复运行的噪声:n=190 时 agree 差约 1 步;跨配置差 ≥3 个点才有意义。
- **机理 capture**:15 个真实 step(8 个 selection-sensitive = dense✓/legacy-tree✗,4 个 control,3 个 Wolfram),在修复后的 vortex server 上逐层 dump 全量 K(fp16)、每个 decode step 的原始 q `[32,128]`、chunk 表、token ids(约 10 GB)。离线对 **9 种打分变体 + oracle** 重新打分,ground truth = full attention 的质量分布。校验:离线重算的 Quest top-64 与 server 实际选择 overlap **1.000**。

**打分变体**(离线全部计算;标 ★ 的可在 server 上跑 end-to-end):

| 变体 | 定义 | 目的 |
|---|---|---|
| ★legacy | q̄ 组平均 + fp8 + corner-max + head-MEAN | TSA default path 的数学复刻 |
| legacy_fp32 | 同上,无 fp8 | 隔离量化 |
| mid | q̄·c | legacy 退化探针 |
| ★quest | 逐 head elementwise + head-MAX | Quest 忠实实现(vortex) |
| quest_hm | 逐 head elementwise + head-MEAN | 隔离 head aggregation |
| ★mixmax | q̄ + elementwise(relu 拆分)+ head-MEAN | 工程混合方案 |
| cent | mean-key + q̄ + head-MEAN | 朴素 centroid |
| cent_hmax | mean-key 逐 head + head-MAX | 隔离 head-MAX |
| ★block | softmax(τ=0.09) + head-MAX | BlockSparse 原版(vortex) |
| ★legacy_py | legacy 数学原样走 fallback 路径 | 用于 §3.4 的 A/B |

mixmax 与 legacy_py 为本研究新实现,均与离线分析数值对拍(mixmax max|Δ|=9.5e-7;legacy_py bit-identical)。

---

## 3. 根因:KV buffer 布局 bug(比打分本身更重要的结果)

### 3.1 buffer 是怎么分配的

每层 KV 被分配成**一个一维 tensor**,每页按 FlashInfer NHD 布局 `[2, page_size, H_kv, D]` 存放(先整块 K,再整块 V):

`python/kv_cache.py:69`
```python
elems_per_page = 2 * page_size * num_kv_heads * head_dim
```
`python/kv_cache.py:105-123`
```python
for layer_id in range(num_layers):
    buf = torch.zeros(
        total_pages * elems_per_page,          # 一维扁平 buffer
        dtype=paged_kv_single[layer_id].dtype,
        device=device,
    )
    ...
    buffers.append(buf)
```

因此,把这个一维 buffer 看成若干行、每行 `H_kv*D` 个元素,**第 r 行对应**:page = `r // (2·ps)`,半区 = `(r // ps) % 2`(0 = K,1 = V),槽位 = `r % ps`。**token t 的 key 所在行是 `r_K(t) = (t // ps)·2·ps + t % ps`**,而不是 t。FlashInfer 的 attention wrapper(`serve.py:545`,`kv_layout="NHD"`)正确理解这个布局,所以模型生成一直是正常的。


**为什么 K、V 会"交错"?为什么要 flatten?SGLang 为什么没事?** 张量在显存里永远是一维线性的,形状只是解释这段内存的元数据。FlashInfer 的 paged 布局 `[num_pages, 2, page_size, H_kv, D]` 规定的内存顺序是"page 0 的全部 K 行、page 0 的全部 V 行、page 1 的全部 K 行、page 1 的全部 V 行……"——这样做是因为 page 是分配和读取的单位,K、V 同页相邻,一条 page-table 项同时覆盖 K 和 V,attention kernel 读一页时 K、V 局部性最好。**`flatten()` 没有改变任何一个元素的位置**,它只是把 5-D 的形状丢掉,把本来就存在的"K 块、V 块、K 块、V 块……"顺序暴露为一维。TSA 选择 flatten 是工程上的便利:自写的 CUDA kernel 只拿裸指针 + 几何参数;`kv_cache.py` 按 `page × elems_per_page` 的偏移做 page 级 memcpy(shared prefix);batched decode 用 `torch.cat` 拼接各请求的 buffer——对任何**知道几何**的消费者(FlashInfer 的 attention 拿到 page_size / H / D 后自己算偏移)都无害。

用 page_size=2、H=1、D=1 画出来:

```
5-D [pages=2, 2, ps=2, 1, 1] 的内存顺序:  p0: K0 K1 V0 V1 | p1: K2 K3 V2 V3
flatten 之后:                          [K0, K1, V0, V1, K2, K3, V2, V3]
selector 按"第 t 行 = token t 的 key"读:   t=0→K0 ✓  t=1→K1 ✓  t=2→V0 ✗  t=3→V1 ✗  t=4→K2 ✗(应是 K4)
```

SGLang 没有这个问题,是因为它的 token pool 把 **K 和 V 放在两个独立的 buffer** 里(`k_buffer[layer]`、`v_buffer[layer]` 各为 `[pool_size, H_kv, D]`),page 只是连续的槽位区间,于是"第 t 行 = 槽位 t 的 key"天然成立,selector 那种下标算术在 SGLang 里是对的;Vortex 又更进一步,连下标都不自己算,由框架按页交付 key 张量。值得一提的是 FlashInfer 的 API 本身两种形式都接受:单个合并的 `[pages, 2, ps, H, D]` 张量,或 (K, V) 两个各为 `[pages, ps, H, D]` 的张量——**如果 TSA 当初选了后者(K、V 分开),flatten 之后 K 行就是连续的、"第 t 行 = token t 的 key"同样成立,selector 的假设会恰好正确。** 所以这个 bug 的精确成因是:TSA 的 serving 选了"K/V 合并按页交错"的那种 FlashInfer 形式,而 TSA 的 selector 按"K 单独成表"的形式写——并且 flatten 让两者在指针层面无法区分(`size(0)` 变成元素总数,越界检查永远通过),错误因此无声。顺带一提,即使不 flatten、直接把 5-D 张量传给 selector,bug 依然在(kernel 拿到的是同一段内存、同一套错误算术),只是 `pool_size` 会变成 `num_pages`、越界检查会把大部分 token 拦掉,表现为另一种错——根因是几何契约不匹配,不是 flatten。




**flatten 是谁的要求?** 是 TSA 自己的选择,FlashInfer 不要求。证据在 TSA 自己的 C++ decode wrapper(`csrc/ts_decode.cu:116-118,178-179`):

```cpp
// k_data can be any shape as long as it's contiguous (we just pass a raw pointer)
// It should be [num_pages, 2, page_size, num_kv_heads, head_dim] flattened or any equivalent shape
...
// NOTE: We assume k_data contains interleaved K/V (FlashInfer format)
// For separate K/V buffers, preprocessing would be needed
paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
    num_kv_heads, page_size, head_dim, batch_size,
    static_cast<c_type*>(kv_data_bf16.data_ptr()),   // 一个裸指针
    page_indices, page_indptr, last_page_len);
```

TSA 把 FlashInfer 作为 C++ 头文件库编进自己的扩展(`3rdparty/flashinfer`,fork 分支 `tree-sparse-buffer-reuse`),调用的是 FlashInfer 的 C++ 结构体 `paged_kv_t`:它只接收**一个裸指针 + 几何参数**(`num_kv_heads, page_size, head_dim`),对张量的"形状"一无所知,5-D 还是 1-D 对它完全等价——所以 "any equivalent shape"。它真正要求的是**内存顺序**:这块连续内存必须按 `[num_pages, 2, page_size, H_kv, D]` 排列,即 K/V 按页交错(这是该接口的单 buffer 形式;TSA 的注释 "For separate K/V buffers, preprocessing would be needed" 说明作者知道 K/V 分开的形式、但没有采用)。于是 `convert_kv_to_interleaved` 里的 `kv_flat = kv_interleaved.flatten()`(`models/qwen3vl_inference.py:570`)纯属 TSA Python 胶水层的便利:一维让 `kv_cache.py` 的页级拷贝(`buf[dst:dst+elems] = src[...]`)、batched 的 `torch.cat` 和预分配扩容(`new[:n] = old`)都变成简单的元素偏移算术——这些用 5-D 沿 page 维同样能做。

所以精确的分工是:**"K/V 按页交错"来自 TSA 选用的 FlashInfer 单 buffer 接口;"flatten 成一维"来自 TSA 自己的胶水代码;"把它当 `[tokens, H, D]` 读"来自 TSA 自己的 selector。** 三者都在 TSA 仓库内,FlashInfer 本身没有做错任何事——它的 decode kernel 拿着正确的几何参数,读的一直是对的。


**各框架的 KV cache 布局对照(2026-08 源码/文档核实)**——"flatten"和"K/V 交错"都不是通行做法,各家各有一套几何,共同点是**所有消费者都通过带形状/stride 的张量或统一的几何参数访问,没有人用裸指针按"token 行"硬算**:

| 栈 | 每层 KV 的形状 | K、V 是否交错 | 是否 flatten | 依据 |
|---|---|---|---|---|
| **SGLang**(token pool) | `k_buffer[layer]`、`v_buffer[layer]` 各 `[pool_size, H_kv, D]` | 否(两个独立 buffer) | 本身就是"按 token 行"的二维表 | `MHATokenToKVPool` |
| **FlashAttention**(`flash_attn_with_kvcache`) | `k_cache`、`v_cache` 各 `(num_blocks, page_block_size, nheads_k, headdim)` + `block_table (batch, max_num_blocks_per_seq)` | 否(两个独立张量) | 否。它的 varlen 接口把一个 batch 的序列沿 token 轴拼成 `(total_k, nheads_k, headdim)` + `cu_seqlens`,这是"拼序列",不是"拼 K/V" | `flash_attn_interface.py` docstring |
| **FlashInfer** | 单张量 `[num_pages, 2, page_size, H_kv, D]`(NHD;每页先 K 块后 V 块),**或** `(k_cache, v_cache)` 两个 `[num_pages, page_size, H_kv, D]` | 两种形式都支持;TSA 选了合并交错的那种 | 否,API 接受多维张量 | FlashInfer paged-KV API |
| **vLLM**(当前 main,FlashAttention 与 FlashInfer backend) | `(num_blocks, num_kv_heads, block_size, 2 * head_size)`,注释 "K and V are packed into the content dim";用 `kv_cache.transpose(1, 2).split(head_size, dim=-1)` 切出 key_cache / value_cache 的**视图** | 是——但在最内层(每个 token 的行 = `[k(D) \| v(D)]`),不是按页交错;旧版本为 `(2, num_blocks, block_size, H_kv, D)`(K/V 在最外层分开) | 否,靠 view/stride | `vllm/v1/attention/backends/{flash_attn,flashinfer}.py` |
| **TSA**(`serve.py`) | FlashInfer 合并形式 `[num_pages, 2, page_size, H_kv, D]` 再 `.flatten()` 成一维 | 是,按页交错 | **是**,且把一维裸指针交给自写 kernel | `convert_kv_to_interleaved`、`kv_cache.py` |

注意 vLLM 的例子最能说明问题:它的 K、V 同样"交错"(每行 K 接 V),但它**从不手算偏移**——`split` 出来的 key_cache 是带 stride 的视图,kernel 拿到的是正确的形状;TSA 的问题不在于"交错",而在于 flatten 之后用一套与几何不符的手写算术去读。


### 3.2 两条 page-selection 路径都把它当成了 `[tokens, H_kv, D]`

调用点(单请求与 batched 两处一样),传入的就是上面的扁平 buffer,`kv_indices = arange(prefill_len)`,即"逻辑 token 位置 = 物理行号":

`serve.py:538-540`
```python
kv_indices = torch.arange(prefill_len, dtype=torch.int32, device=device)
for lid in range(num_layers):
    selector.compute_centroids(paged_kv[lid], kv_indices, lid)
```


**一个容易误读的细节:`k_buffer` 不是独立存储的 K 张量,只是 selector 形参的名字。** `compute_centroids(self, k_buffer, kv_indices, layer_id)` 的签名和 docstring(`k_buffer: Key cache buffer [pool_size, num_kv_heads, head_dim]`)表达的是 SGLang 风格的"给我一张 key 表";但 TSA 的 serving 路径里**根本不存在**一个只含 K 的张量:`convert_kv_to_interleaved` 用 `torch.stack([key_pages[0], value_pages[0]], dim=1)` 把每层的 K、V 合成一个张量再 flatten,`kv_cache.py` 每层只分配一个同时装 K 和 V 的 `buf`,`serve.py` 把 `paged_kv = kv.buffers` 的 `paged_kv[lid]` 原样填进 `k_buffer` 这个形参。§3.3 真实路径测试打印的 `numel = total_pages * 2 * ps * H * D` 里的那个 **2** 就是 K 和 V 都在里面的证据。换句话说,这个 bug 用一句话就能概括:**函数说"给我 key 表",调用方给的是"整块 K+V 页内存",而参数名让两边都以为自己是对的。**(分析脚本和 §3.3 的第二个测试传的才是真正的 3-D key 表 `[tokens, H_kv, D]`,所以修复代码里才有 `if k_buffer.dim() == 1` 这个分支。)


**PyTorch fallback(修复前,`python/tree_sparse_selector.py.bak_predump:383-391`)**:
```python
for chunk_id, chunk in enumerate(self.chunks):
    logical_range = torch.arange(chunk.start_idx, min(chunk.end_idx + 1, len(kv_indices)), ...)
    physical_locs = kv_indices[logical_range]
    chunk_keys = k_buffer[physical_locs]          # 一维 tensor 用 token 下标索引 → 取到的是标量!
    if len(chunk_keys) > 0:
        ck = chunk_keys.float()
        chunk_maxs[chunk_id] = ck.max(dim=0).values   # 标量 max 广播成 [H_kv, D]
        chunk_mins[chunk_id] = ck.min(dim=0).values
```
在一维 buffer 上 `k_buffer[physical_locs]` 取到的是第 t 个**标量元素**(几乎全部落在 page 0 的前几行里),`max/min` 之后是一个标量广播成整个 envelope——每个 chunk 的 envelope 完全是噪声。所有 `TSA_SCORING_IMPL=vortex` 的运行(08-14 在线 tsa_B4096、online-vx block、离线 vx_*)都走这条路,**全部作废**。08-15 修复:

`python/tree_sparse_selector.py:407-413`(当前)
```python
if k_buffer.dim() == 1:
    _ps = self.config.page_size
    k_rows = (k_buffer
              .view(-1, 2, _ps, self.num_kv_heads, self.head_dim)[:, 0]   # 只取每页的 K 块
              .reshape(-1, self.num_kv_heads, self.head_dim))             # → [tokens, H_kv, D]
else:
    k_rows = k_buffer
...
chunk_keys = k_rows[physical_locs.long()]
```

**CUDA selector(TSA default path,至今未修)**:wrapper 直接把一维 tensor 的 `size(0)`(= 总元素数)当 `pool_size`,原样把指针交给 kernel:

`csrc/ts_pybind.cu:235-261`
```cpp
void compute_envelopes(torch::Tensor k_buffer, torch::Tensor kv_indices, int layer_id, int num_kv_heads, int head_dim) {
    ...
    int pool_size = k_buffer.size(0);      // 一维 tensor → 这是元素总数,不是行数
    ...
    launch_compute_envelopes(k_buffer.data_ptr(), kv_indices.data_ptr<int32_t>(), ..., pool_size, ...);
```
kernel 按 `[pool_size, H_kv, D]` 的行布局寻址:

`csrc/ts_tree_sparse.cu:53-86`
```cpp
__global__ void compute_envelopes_batched_kernel(
    const T* __restrict__ k_buffer,          // [pool_size, num_kv_heads, head_dim]   ← 注释里的假设
    const int32_t* __restrict__ kv_indices,  // [seq_len]
    ...
    for (int pos = start_idx; pos <= end_idx && pos < pool_size; pos++) {
        int physical_loc = kv_indices[pos];
        if (physical_loc >= 0 && physical_loc < pool_size) {
            int k_idx = physical_loc * num_kv_heads * head_dim +     // 第 physical_loc 行
                        head_id * head_dim + feat_idx;
            ...
            val = __bfloat162float(k_buffer[k_idx]);
```
于是 token t 读到的是扁平 buffer 的**第 t 行**,而它的 key 在第 `(t//ps)·2ps + t%ps` 行。代入 §3.1 的行映射:

| token t 的范围 | kernel 实际读到的 | 正确与否 |
|---|---|---|
| `[0, ps)` | page 0 的 K 块,槽位 t | ✅ |
| `[ps, 2ps)` | page 0 的 **V** 块 | ❌ 读到 value 向量 |
| `[2ps, 3ps)` | page 1 的 K 块 = token `ps..2ps-1` 的 key | ❌ 错了一页 |
| `[3ps, 4ps)` | page 1 的 V 块 | ❌ |
| 一般 t | page `t//(2ps)`,半区 `(t//ps)%2` | 只有第一页正确 |

**结论:在 TSA default path 上,除了 context 开头的 `page_size` 个 token,所有 chunk 的 envelope / centroid 统计量都由错误的向量算出——一半 chunk 用的是 V 向量,另一半用的是错页的 K。** 打分数学本身(corner-max 等)根本没机会发挥作用。`compute_centroids` 的 CUDA kernel 寻址相同,centroid 模式同样受影响。本报告此前版本(及 08-15 的 fallback 修复说明)写过"CUDA 路径不受影响(kernel 理解布局)"——**那是错的**,kernel 的注释就写明它假设行布局。

### 3.3 测试复现(两个测试,spark00 实跑)

**测试 1(决定性):真实 serving 路径 + HF 张量作 ground truth**(`scoring_case_study/test_cuda_layout_realpath.py`)。构造 HuggingFace 格式的 K/V(1 层,H_kv=4,D=128,512 个 token,小噪声),把 token 100 的 key 设为 8·u(u 为单位向量),query 全部 head 取 u;然后**原样调用 serve.py 用的两段真实代码**——`convert_kv_to_interleaved`(`models/qwen3vl_inference.py`)和 `build_shared_prefix_kv`(`python/kv_cache.py`)——得到 serving 时 selector 真正拿到的那个 buffer,再喂给真实的 CUDA selector(fixed16 chunk,top-k=1,关闭 always-include)。测试本身不对布局做任何假设:它在产出的 buffer 里**搜索**每个 HF 向量实际落在哪一行。输出:

```
serve.py buffer: dim=1 numel=540672 (= total_pages*2*ps*H*D -> 33 pages)
HF key of token 100 is stored at flat row 196 (max|err|=0.00e+00); selector reads row 100 for token 100
flat row 100 actually holds: ['K of token 52']
correct chunk = 6; if the selector indexes rows by token id, it will credit the needle to chunk 12
CUDA selector (TSA default path)    use_cuda=True  -> picked chunk/page 12
PyTorch fallback + layout fix       use_cuda=False -> picked chunk/page 6
VERDICT: BUG CONFIRMED on the real path: CUDA selector picked chunk 12 != correct 6;
         12 is exactly the chunk that contains flat-row index 196 (row-index-by-token-id prediction)
```
三个测得的事实互相锁死:token 100 的 key 在第 196 行(= (100//16)·2·16 + 100%16,§3.1 的公式);selector 为 token 100 读的第 100 行装的是 token 52 的 key(page 3 的 K 块第 4 槽);CUDA selector 选了 12 = 196//16——needle 的信号被记到了"行号当作 token 号"时所属的 chunk 上。

**测试 2(机理版):合成 buffer**(`scoring_case_study/test_cuda_layout.py`)。256 token、page_size=16,K 行填 token id、V 行填 −1000,query 全 1。正确答案 chunk 15;CUDA selector 在扁平布局上选 **14**(读到 page 7 的 K,而 chunk 15 读到 V 块的 −1000),同一 kernel 喂 3-D 行布局时选 15,修复后的 fallback 选 15 且 16 个 envelope 精确等于 `[15, 31, …, 255]`——说明 kernel 在行布局下本身正确,错在调用点传的布局。

### 3.4 end-to-end 验证:同一套数学,换路径,4.2×

`lpy_B4096v2` 把 legacy 数学(q̄ 组平均 + fp8 + corner-max + head-MEAN,与离线分析 bit-identical)搬到**修复后的 fallback 路径**,fixed16 k256。前 44 步(0 错误)与历史 TSA default path 结果在**完全相同的 29 个 index 步**上配对:

| 配置 | agree / 29 |
|---|---|
| dense | 19 (65.5%) |
| legacy 数学 × 修复后路径(`lpy_B4096v2`) | **17 (58.6%)** |
| Quest 忠实 × 修复后路径(`vx2_quest`) | 17 (58.6%) |
| BlockSparse 忠实 × 修复后路径(`vx2_block`) | 19 (65.5%) |
| **legacy 数学 × TSA default CUDA path** | **4 (13.8%)** |

legacy 数学与 Quest 忠实打分逐分打平;唯一的差别是布局 bug。corner-max / head-MEAN 这些数学偏差在 B4096-fixed16 的 end-to-end 上代价 ≈ 0。

此前困扰我们的"覆盖悖论"(§5.4:legacy 数学的 per-layer needle 覆盖优于 BlockSparse,end-to-end 却差一倍)也由此解释:离线机理分析用的是 dump 出的**正确** K,而 server 上的 TSA default path 从未见过正确的 K。

### 3.5 修法与影响面

- **修法(一行)**:在 `ts_pybind.cu` 的 `compute_envelopes/compute_centroids` 入口,或在 `serve.py` 调用前,对一维 buffer 做与 fallback 相同的 `view(-1, 2, ps, H_kv, D)[:, 0].reshape(-1, H_kv, D)`,再传给 kernel(kernel 本身无需改动,已由 §3.3 第二行验证);或给 kernel 加 `2·page_size` 的行 stride 参数。**已于 2026-08-22 在 spark00 容器的 `tree_sparse_selector.py::compute_centroids` 中实现**(K-rows 视图在 CUDA 分支之前计算,`contiguous()` 后传给 CUDA selector;备份 `.bak_precudafix`):`test_cuda_layout_realpath.py` 重跑后 CUDA selector 选中正确的 chunk 6(修复前为 12)。仍未提交到 upstream。
- **影响面**:TSA 从建库起所有走 default path 的评测——6 月 k32 研究(0.41% 在线成功率、"88% element-index grounding 失败")、8 月初 3-way 报告全部 legacy 数字、本报告 §4 全部 "legacy" 行——都需要在修复后重跑。它们测量的不是打分方法,而是这个 bug。
- **为什么一直没被发现**:生成文本始终正常(attention 走 FlashInfer,布局正确);选择质量只体现在 agree rate 上,而 legacy 的 18–29% 被解释成了"sparse attention 的代价";always-include recent/first 提供了保底,让结果看起来"合理但偏低"。

### 3.6 bug 的来源(git 考古):从仓库第一个 commit 起就在,不是我们引入的

| 代码位置 | 引入 commit | 日期 | 作者 | 内容 |
|---|---|---|---|---|
| `csrc/ts_tree_sparse.cu`:kernel 注释 `k_buffer // [pool_size, num_kv_heads, head_dim]` 与 `k_idx = physical_loc * num_kv_heads * head_dim + …`(初始版 L54、L90) | `032877b` | 2026-04-08 | Jiaheng Lu | "Initial implementation of TreeSparse attention with CUDA kernels" |
| `csrc/ts_pybind.cu`:`pool_size = k_buffer.size(0)`(初始版 L58) | `032877b` | 2026-04-08 | Jiaheng Lu | 同上 |
| `python/tree_sparse_selector.py`:docstring `k_buffer: Key cache buffer [pool_size, num_kv_heads, head_dim]`、fallback `chunk_keys = k_buffer[physical_locs]`(初始版 L217、L258) | `032877b` | 2026-04-08 | Jiaheng Lu | 同上 |
| `models/qwen3vl_inference.py:convert_kv_to_interleaved`:返回 "List of **flattened 1D** tensors per layer",格式 `[num_pages, 2, page_size, num_heads, head_dim]`(K/V 按页交错) | `032877b` | 2026-04-08 | Jiaheng Lu | 同上 |
| `benchmark_tpot.py:178`:`selector.compute_centroids(paged_kv[layer_id], kv_indices, layer_id)`——把上面那个扁平交错 buffer 直接喂给 selector | `032877b` | 2026-04-08 | Jiaheng Lu | 同上 |
| `python/kv_cache.py:105-109`:batched serving 的扁平分配 `total_pages * elems_per_page` | `fee15fb` | 2026-04-15 | Jiaheng Lu | "Batch inference support with DirectDecode and shared-prefix KV cache" |
| `serve.py:538-540`:serving 调用点,沿用 benchmark 的写法 | `fe4690a`(首版)/ `2961825b`(现行) | 2026-05-20 / 05-25 | JhengLu | "update the tpot-no-share mode" |
| 当前 blame 指向的 `2867300f`(2026-07-18,JhengLu) | — | — | — | 对同一批行的后续重写,语义未变 |

也就是说,**在仓库的第一个 commit 里,selector 就已经按 `[pool_size, H_kv, D]` 的行布局写成,而同一个 commit 里喂给它的 `paged_kv[layer_id]` 就已经是 FlashInfer 的扁平交错页布局**——两者从第一天起就不匹配。之后 `kv_cache.py`(04-15)、`serve.py`(05-20)都沿用了这个调用方式。本项目(Shiqi He)在该仓库的 3 个 commit(`0e1b36c` 06-19 serving 的多模态 prefill/device_map,以及两次 merge)均未触及这些行;我们 08-15 对 fallback 的修复是这条路径上第一次被修正。

**影响范围**:TSA 在这套代码上产出的**所有**使用默认 CUDA selector 的数字——包括早于本项目的 benchmark——其 page 选择统计量都是在错误向量上算的。它能存活四个月的原因:① 生成文本始终正常(attention kernel 走 FlashInfer,布局正确);② `pool_size = k_buffer.size(0)` 对一维输入是一个巨大的数,kernel 里的越界检查永远不触发,没有任何报错;③ 没有把 selector 的输出和一个参考实现对拍过的单元测试——§3.3 的 `test_cuda_layout.py` 是第一个。

**upstream 现状核查(2026-08-21,通过 GitHub API 直接读 `agentic-browser-project/TreeSparseAttention`)**:main 分支 HEAD `0457e95`(2026-08-20 推送)上五处代码**原样未动**——`serve.py:532-534/754-756` 仍把 `paged_kv[lid]` / `kv.buffers[lid]` 与 `kv_indices = arange(prefill_len)` 传给 `compute_centroids`;`csrc/ts_pybind.cu:179,246` 仍是 `pool_size = k_buffer.size(0)`;`csrc/ts_tree_sparse.cu:54/81`(envelope kernel)与 `:105/141`(centroid kernel)仍按 `[pool_size, num_kv_heads, head_dim]` 行布局寻址;`python/tree_sparse_selector.py:367,394` 仍是 `k_buffer[physical_locs]`(**连 fallback 的修复也不在 upstream**,目前只存在于 spark00 容器的工作树);`python/kv_cache.py:69,106` 的扁平分配未变。仓库另外两个分支 `A100`(HEAD 04-29)、`chunk_sim`(HEAD 05-01)比 main 更老,后者同样含有行布局寻址且无修复。**结论:截至 2026-08-20 的最新 main,这个 bug 没有被修过。**

两条相关的 upstream 新提交值得知道:① `62919d6`(08-20,"cheap scorer ablation (F) -- per-dim bound doubles recall"):作者在 CPU 上离线对比 centroid / 2-corner / per-dim envelope bound,结论 "the per-dim envelope bound … ~doubles recall@100 over the current 2-corner (23%→38% at B=50, 43%→59% at B=128) at the same cost — a free one-line scorer upgrade"——与本报告 corner-max → 逐维 max(mixmax)的结论独立吻合;该实验在 CPU 离线进行,不经过 serving 的 selector 路径,因此不受本 bug 影响。② `0457e95`(08-20,"graded WebVoyager pass rates: dense 43.1% / block_sparse 34.9% / quest 34.8%"):跑在 **vortex+sglang** 栈、Qwen3-32B 上,不是 TSA 的 `serve.py`,因此也不受本 bug 影响——但这意味着 upstream 至今没有一组 TSA 自身 selector 路径上的正确数字。

**为什么这个 bug 只在 TSA 出现、Vortex 没有?** 本质不是"kernel 和 python 不一致"——TSA 的 CUDA kernel 和 PyTorch fallback 彼此是一致的(都假设 `[tokens, H_kv, D]` 行布局)。不一致的是 **selector(两个实现)与 serving 栈其余部分对"KV cache 长什么样"的两套约定**:

| 约定 | KV 怎么存 | 取 token t 的 key | 谁在用 |
|---|---|---|---|
| **token-pool**(SGLang 风格) | K、V 各一个 buffer,每个 `[pool_size, H_kv, D]`,一行一个物理 token 槽位;`req_to_token` 表把逻辑位置映射到槽位 | `k_buffer[slot]` | TSA selector 的 API(docstring `Key cache buffer [pool_size, num_kv_heads, head_dim]`,参数名 `req_to_token` 都是 SGLang 词汇) |
| **paged**(FlashInfer / vLLM 风格) | K、V 按页合存:`[num_pages, 2, page_size, H_kv, D]`,每页先 K 块再 V 块;page table 映射逻辑页→物理页 | `kv[page, 0, slot]` | TSA 的 serving(`convert_kv_to_interleaved` + `kv_cache.py` + FlashInfer decode wrapper),因为 FlashInfer 的 attention kernel 要求这个布局 |

两种约定各自都是标准做法;TSA 的 selector 说的是前一种方言,TSA 的 serving 说的是后一种,同一个仓库里两段手写代码对同一块显存持不同理解。把 5-D 的 paged tensor `flatten()` 成一维之后,对一个只拿指针的 kernel 来说两种布局看起来一模一样(一个指针 + 一个 `pool_size`),而 `k_buffer.size(0)` 对一维 tensor 是元素总数,越界检查永远通过——于是没有任何一层报错。

Vortex 不会有这个问题,是因为它**运行在 SGLang 内部**:算法代码(`forward_cache(cache["k"], loc=…)`,page-packed 的 `[S, 1, D]`)从框架手里拿到的就是"按页整理好的 key 张量",由拥有布局的框架负责索引,算法从不自己对原始显存做下标运算——布局知识只在一处(框架的 indexer/loader),不存在"两段代码对同一块内存各有理解"的机会。TSA 则自己重造了 cache(自己把 HF KV 转成 FlashInfer 页 + 自己写 selector 的下标算术),两段都是手写,同一个初始 commit 里同时诞生,之后从未交叉校验。所以这不是"TSA 对 dimension 的处理与传统不同",而是 TSA 把两种传统各用了一半。

**是 SGLang 的 bug 还是 TSA 的?** 是 TSA 的。上表每一个文件(`csrc/ts_tree_sparse.cu`、`csrc/ts_pybind.cu`、`python/tree_sparse_selector.py`、`python/kv_cache.py`、`models/qwen3vl_inference.py`、`serve.py`)都在 TreeSparseAttention 仓库里、由 TSA 作者编写;`serve.py` 是一个独立的 HuggingFace-transformers + FlashInfer 服务,**SGLang 完全不在这条路径上**(本项目 8 月初试过的 vortex/SGLang 环境是另一条已放弃的线,与此无关)。FlashInfer 只提供 attention kernel 和它文档化的 paged 布局 `[num_pages, 2, page_size, H, D]`——TSA 自己的转换代码正确地生成了这个布局,FlashInfer 的 attention 也正确地读了它;出错的是 TSA 自己写的 selector kernel 对输入布局的假设(`[pool_size, H_kv, D]` 行布局)。一个可能的由来(推测,非事实):selector 的 API 带有 `req_to_token` 参数、docstring 写的是 SGLang 风格的 token pool 形状 `[pool_size, num_kv_heads, head_dim]`,像是按 SGLang 的 `token_to_kv_pool.k_buffer` 设计的;但在这个仓库里它从第一个 commit 起接到的就是 FlashInfer 布局,两者从未对上。

### 3.7 方法论教训

- 机理指标不能跨 chunk geometry 搬运(§5.2):我们曾据 tree 上的 needle hit rate 预测 fixed16 end-to-end 15–22%,实际 52.1%。
- budget accounting 必须写明单位(§5.3)。
- 基础设施:`pgrep -f` 自匹配造成两次假阳性;日志里的 NUL 让 `grep` 静默跳过(需 `-a`);repeat 噪声要先测再解释跨配置差。

---

## 4. end-to-end 结果矩阵

### 4.1 B4096 主矩阵(half1,190 个 index 步,全部 errors=0)

| 打分 \ chunk geometry | fixed16(k256) | tree 变长(k64) |
|---|---|---|
| dense 上限 | **51.1%** (97) | — |
| Quest 忠实(修复后路径) | 51.6% (98) | 46.8% (89) |
| BlockSparse 忠实(修复后路径) | **52.1% (99)** | 运行中(§9) |
| mixmax(修复后路径) | 50.5% (96) | 47.9% (91) |
| legacy 数学 × 修复后路径 | 58.6% @ n=29 配对(运行中) | 排队(§9) |
| **legacy × TSA default path(布局 bug)** | 20.5% (39) | 28.9% (55) |
| legacy centroid × TSA default path(布局 bug) | 18.4% (35) | — |

- **fixed16 + 修复后路径 = dense**:三个忠实打分与 dense 挤在一簇(±1 个点);Quest 与 BlockSparse 有 **144/190 步输出逐字相同**,分歧 ≈ 重复噪声。我们称之为 **transparent regime**:budget 足够、page 足够细时,打分函数对结果不可见。
- **tree 一致低 4 个点**,忠实打分之间同样无差别——差的不是打分,是 chunk geometry(§5.3)。
- bug 行 ≈ 随机选页水平(对照:修复前的噪声选择在线仍有 26% 任务成功)。

### 4.2 verdict 组成

| 配置 | agree | valid | invalid | none |
|---|---|---|---|---|
| dense | 97 | 154 | 0 | 36 |
| vx2-tree | 89 | 151 | 4 | 35 |
| vx2-quest | 98 | 151 | 1 | 38 |
| mm-tree | 91 | **157** | **0** | 33 |
| legacy-tree(bug) | 55 | 134 | **13** | 43 |
| legacy-quest(bug) | 39 | 132 | **12** | 46 |
| legacy-block(bug) | 35 | 136 | **16** | 35 |

dense 本身有 19% 的 none(text-only A3B 复现多模态 omni 轨迹的天花板)。修复后路径的 invalid 0–4 ≈ dense;bug 路径 invalid 12–16 = 幻觉 index(选择集里没有证据,模型硬编)。

### 4.3 budget 维度

B8192(bug 路径):tree 43.7 / quest 47.4 / block 47.9;B16384:tree 51.1(= dense)。**修复后路径的 B2048**:fixed16 + Quest k128 = **47.9%**(91/190,0 错误,2026-08-22)——只比 B4096 低 3.7 个点、仍贴近 dense;fixed16 的 transparent regime 一直延伸到 2048 token(我此前预测 40–46%,偏低)。结合机理饱和曲线(§5.2):**B8192 是所有打分 × 两种 geometry 的机理饱和点**。旧观察"budget 大到 8192 就洗掉一切差异",实际是 budget 大到连垃圾统计量也能覆盖 needle。修复后,4096 即达 dense;B2048 的 end-to-end(vxq/vxt)运行中。

### 4.4 分类别(全表 `per_category_scoring.md`)

修复后逐类别恢复:Wolfram bug-tree 1/7 → 修复后 tree **6/7(= dense)**;ESPN 3→9(= dense);Allrecipes 3→9;GitHub 4→8;Huggingface 上 vx2-quest 6/18 超过 dense 的 4/18。

---

## 5. 机理:两类打分各自的病(全部在正确 K 上测量)

### 5.1 needle 与 mass 是两个不同目标

| (tree k64) | needle hit@64 | needle 平均 rank | 与真实 mass 的 rank 相关 | mass recall@64 | 选中 chunk 均长 |
|---|---|---|---|---|---|
| mixmax | **.895** | **34.9** | **.890** | .842 | 108 tok |
| quest | .873 | 36.0 | .844 | .600 | 108 |
| legacy(数学) | .720 | 52.7 | .699 | **.913** | 65 |
| mid | .603 | 70.2 | .459 | .868 | 45 |
| cent | .612 | 71.4 | .388 | .848 | 44 |
| cent_hmax | .516 | 86.3 | .233 | .793 | 46 |
| block | .478 | 98.3 | .183 | .819 | 35 |

决定 end-to-end 的是 needle 列:attention mass 被 BOS sink 与 recent tokens 主导,而那部分由 `always_include_first/recent` 保证。legacy 的高 mass recall 含 sink 的白送分。

### 5.2 sweep:budget × geometry × 打分(n=15 全量,`sweep_v3.md`)

needle hit rate 节选:

| 变体 | tree k32 | tree k64 | tree k128 | fixed16 k128 | fixed16 k256 | fixed16 k512 |
|---|---|---|---|---|---|---|
| legacy | .446 | .720 | .917 | .528 | .730 | .963 |
| quest | .655 | .873 | .998 | .410 | .646 | .963 |
| mixmax | .649 | .895 | 1.000 | .482 | .728 | .976 |
| cent | .356 | .612 | .835 | .564 | .760 | .976 |
| block | .225 | .478 | .743 | .553 | **.764** | .957 |
| oracle | .591 | .899 | .994 | .550 | .754 | .958 |

1. **家族排序随 geometry 反转**:tree 上 envelope ≫ centroid(.87 vs .48),fixed16 上 centroid ≥ envelope(.764 vs .646)且全员贴近 oracle。centroid dilution 随 chunk 变大加重;head-MAX 的噪声随 page 数增多加重(fixed16 约 540 页、每页只有 16 个 key,32-head MAX 的单 head 假阳性激增,quest 的 rank 相关掉到 .585 全场最差;head-MEAN 版恢复到 .730)。
2. **饱和点**:tree 的 envelope 家族 k128(= B8192)饱和;fixed16 全员 k512(= B8192)饱和。per-layer hit 只有 .65–.90 时 end-to-end 已无损——模型只需在部分层拿到证据(§5.4)。
3. **低 budget 下 tree 反超**:tree k32 的 envelope .649–.655 > fixed16 k128 的 .41–.48;end-to-end 验证 = vxt_B2048 / vxq_B2048(运行中)。

### 5.3 budget accounting:envelope 在 tree 上的优势有一部分是多读 token 买来的

quest/mixmax 在 tree 上选中的 chunk 平均 108 token(k16 时 240;语料均长 39),legacy 65:**同为 top-64 chunk,envelope 家族实际读约 6.9k token,legacy 约 4.2k**——width bias = 隐性 budget 膨胀。按 token 对齐后,tree 上 ≤B4096 的机理赢家反转为 q̄·c 均值家族(B4096-token hit:cent .748 / mid .739 / legacy .677,而 mixmax .142 / quest .133)。含义:对速度(KV 读取字节数)诚实的比较必须 token 对齐;fixed16 天然免疫(chunk 数 ≡ token 数)。

### 5.4 跨层 union 分析:为什么低 per-layer hit 不一定输

page 选择每层独立发生。360 个 (step, 请求) 的层覆盖分布(tree):

| 变体 | 平均层覆盖 | ≥50% 层命中的步 | 任一层命中 | 全层 miss |
|---|---|---|---|---|
| mixmax | .895 | 89% | 99.7% | 1/360 |
| quest | .873 | 88% | 98.3% | 6/360 |
| legacy(数学) | .719 | 81% | 100% | 0/360 |
| cent | .612 | 63% | 94.4% | 20/360 |
| block | .478 | 50% | 86.7% | 48/360 |

miss 在层带上均匀分布。BlockSparse-on-tree end-to-end 不塌的机制:一半的步仍有半数层覆盖,行为阈值远低于"多数层命中";风险集中在 13% 的全层 miss 步。而 legacy **数学**的覆盖优于 block 却 end-to-end 崩溃——正是 §3 的 bug(机理用正确 K,server 用错误 K)。

### 5.5 宽度归一化 α 扫描(离线 dumps,n=15)

mixmax 的宽度项改为 `|q̄|·w·(16/L_chunk)^α`,tree:

| α | k64 hit | B4096-token hit | B2048-token hit | 选中均长@k64 | rank |
|---|---|---|---|---|---|
| 0(= mixmax) | **.885** | .179 | .085 | 108 | **34.5** |
| **0.25** | .749 | **.805** | **.641** | 56 | 49.3 |
| 0.5 | .417 | .791 | .613 | 27 | 100.6 |
| 1.0 | .312 | .738 | .568 | 17 | 140.8 |
| (oracle) | .898 | .511 | .391 | 100 | 29.6 |

宽度项本身就是 needle 信号(chunk-count budget 下 α=0 最优,α≥0.5 摧毁检索);但 **α=0.25 在 token budget 下全面胜出**(B4096-token hit .805,超过均值家族与 mass-oracle),选中均长压回 56。规则:chunk-count accounting 用 mixmax(α=0);token accounting 用 α≈0.25(需要 selector 支持按累计 token 的贪心选择)。

---

## 6. Case Studies(全部来自真实轨迹步;完整版 `cases.md` / `cases_centroid.md`)

### 案例 1 | Coursera--8:head 平均 + corner-max 联手淹没搜索框

参考元素 `[54]<input placeholder="What do you want to learn?">` 位于 138-token 的 `<form role=search>` 子树 chunk(P=291)。L24 分解:Quest 的最强 head(head 24)给出 q·c=60.3、**|q|·w=600.0** → rank 27;legacy 在同一 chunk 上,8-head 平均后 **q̄·c 变负(−2.67)**,corner 项抵消成噪声(0.79)→ rank 113。指向表单的那一个检索 head 被 7 个无关 head 投票压倒,corner-sum 再把残余信号抵消。输出 index 数字的 decode steps 上 Quest rank 1–12。end-to-end:修复后配置 ✓54(= dense)。

### 案例 2 | ESPN--33:同机制 + centroid 的时间漂移

参考元素 `[77]<input id=global-search-input>`(138-token 导航子树,P=378,树碎成数百个 4-token 小链)。centroid 的 rank **在 decode 过程中从 64 漂到 275**(query 漂移主导了低区分度的 centroid 分),Quest 稳定在 19–25(M/m 是与 query 无关的极值)。legacy 数学 rank 71,刚好在 64 的 cutoff 外侧。

### 案例 3 | centroid dilution 的定量解剖

Coursera 同 chunk:**centered cosine(centroid, 元素 key)= 0.708,chunk 内最佳单 token = 0.932,33% 的 token 与背景不可区分**——chunk 包含元素,均值不包含。centroid 分 −2.20 落在 291 个 chunk 的中位数附近。**91% 的检索分来自 |q|·w 宽度项,正是 mean-pooling 在打分前就销毁的分量**;head-MAX/softmax 都救不回。同型:ESPN(0.642/0.930)、Allrecipes(0.739/0.935)、BBC(46-token 混合 chunk,0.761/0.908)——dilution 跟的是异质度,不是长度。

### 案例 4 | Wolfram:从 14% 到 86% 的逐层机理

legacy 数学把 textarea / Compute 按钮排在 39–62,骑在 k=64 的 cutoff 上,且 L24–39 有中层下沉(hit 掉到 .47–.51);Quest 稳定 24–29、逐层平坦。end-to-end:bug-tree 1/7 → 修复后 tree **6/7 = dense**。(chunking 健康:P=132–218,p50 10–16 token,覆盖 100%,"chunk accounting artifact" 的猜想被否定。)

### 案例 5 | 反例:width bias 埋掉小元素,centroid 反而赢

Wolfram--10 的 `[104]<button>Accept`(11-token cookie chunk):centroid = 去噪后的元素(centered cosine 0.917,高于最佳单 token 的 0.852),cent rank 51 / hit .71;Quest 的 q·c 全场最高(97.9),却因 11-token 窄 envelope(|q|·w=286)竞争不过 236–256-token 大 chunk(600+)→ rank 99 / hit .05;Quest top-8 全是 256-token 大块,corr(尺寸, 分数)=0.73。Cambridge--26 的 21-token onetrust 按钮同型。**centroid 唯一的赢面 = 小而同质的 chunk,恰是 envelope width bias 的盲区——两种失败模式互补。**

### 案例 6 | softmax 是负资产(BlockSparse 原版逐项检验)

17,280 行普查:cent 命中而 block miss **2,652** 行,反向仅 338。head-MAX 单独 −9.6 个点,softmax 再 −3.8。机制:数百页上近平坦的 head 给出 ≈1/P 的均匀概率,head-MAX 锁定在"最尖锐"的 boilerplate head 上;多个 head 温和支持(head-MEAN 的强项)被逐 head 分布掩埋。BlockSparse 原版在 tree 上严格差于朴素 centroid;它在 fixed16 上的 52.1% 靠的是细 page 消除 dilution + transparent regime,而非 softmax/head-MAX 有功。

### 补充 | Amazon--2(P=628):budget 本身成为约束

context ≈ 20k、628 个 chunk 时 top-64(10%)下所有打分欠覆盖,模型输出 none——context 超过 ~16k 后 k64 的固定 chunk 数而非打分成为主约束(tree 的 chunk 数随 context 线性增长,budget 应随之伸缩)。

---

### 6.9 B4096 discordant step 的 dump 复盘(2026-08-23):选对了 chunk,错在决策

对 half1 上新方法与 fixed16-Block 结果不一致的 10 个 step,用新方法配置(tree_min16 + mixmax_wn α=.25 + 4096-token,batch 1)重跑并保存全层 K、每个 decode 步的 q、每层实际选中的 chunk(`dumps_discord10/`,分析器 `case_discord10.py`,结果 `case_dumps/case_discord10_B4096.md`)。

| step | 原 leg 结果 → batch-1 重跑 | ref 所在 tree chunk(大小) | 实际被选中的比例(全部 decode 步 / 最后 15%) | 同一 query 下离线 Block@256 页 / Quest@256 页命中 | 选错的元素在哪 |
|---|---|---|---|---|---|
| Allrecipes--3 s2 | 81117(幻觉)→ **8117 ✓** | #66(72 tok) | 0.88 / 0.87 | 0.84 / 0.65 | — |
| Apple--0 s2 | 492 → **493 ✓** | #47(95 tok) | 0.92 / 0.88 | 0.87 / 0.77 | 同一 chunk |
| ESPN--33 s8 | none → **15436 ✓** | #73(178 tok) | 0.66 / 0.77 | 0.79 / 0.37 | — |
| Allrecipes--25 s11 | 31448 → 31448 ✗ | #84(27 tok) | **0.99 / 1.00** | 0.97 / 0.84 | 相邻 chunk #83(162 tok,选中 74%) |
| ESPN--33 s10 | 15474 → 15474 ✗ | #95(178 tok) | 0.57 / 0.70 | 0.79 / 0.42 | **同一 chunk** |
| GitHub--22 s1 | 1599 → 58 ✗(dense 也错) | #48(96 tok) | 0.98 / 0.99 | 0.95 / 0.58 | **同一 chunk** |
| level1-5 s2 | 1645 → 1645 ✗(dense 也错) | #60(68 tok) | 0.90 / 0.83 | 0.73 / 0.57 | **同一 chunk** |
| Coursera--8 s10 | 22287 → 22294 ✗(dense 也错) | #79(17 tok) | 0.55 / 0.54 | 0.37 / 0.42 | 相邻 chunk(mass rank 98:模型本来就不看它) |

三个结论:
1. **5 个"真损失"里 3 个在同配置重跑后变对**(Allrecipes--3、Apple--0、ESPN--33 s8)——它们是 decode 层面的随机波动(batch 组成 / 数值非确定性),不是选择器的错。这是 B4096"噪声带"结论的直接证据。
2. **剩下的错误都不是 needle 没选中**:ref 所在 chunk 在 57–99% 的 decode 步里被选中(最后 15% 步里 70–100%),而模型选错的元素与 ref 在**同一个 chunk**(GitHub、level1-5、ESPN s10)或相邻 chunk。即上下文在手,决策出错——dense 在其中 3 个也错。
3. 同一 query 下离线对比,新方法的 mass recall(0.80–0.95)全面高于 Block(0.61–0.83),needle 命中也不低于 Block;可 online 两者打平——再次说明 B4096 以上 selector 的质量已经不是瓶颈。

区域分配(实际选择,按 token 计):小页面(T≈6–8k)DOM 覆盖 78–96%、system prompt 36–62%;长 episode(ESPN,T=15–23k,其中 5.7–13k 是历史步骤)DOM 40–46%、历史 9–16%、system prompt ~20%。

### 6.10 B2048 真损失 step 的 dump 复盘(2026-08-23):紧预算下 tree 的两个失效机制

对 strat20 上 B2048 tree 的 9 个真损失 + 4 个真赢 step,用 tree_min16 + mixmax_wn + **2048**-token 配置(batch 1)重跑并 dump(`dumps_discord2048/`,结果 `case_dumps/case_discord2048_B2048.md`)。与 B4096 的复盘形成鲜明对比:

| step(原结果 → 重跑) | ref 所在 chunk 大小 | **ref chunk 实际被选中比例** | 同 query 离线 Block@128 页 / Quest@128 页 | 选错的元素被选中比例 |
|---|---|---|---|---|
| Coursera--8 s1(none → 291 ✗) | **138 tok**(search input 子树) | **0.01**(rank 中位 93/135) | 0.24 / 0.21 | 0.41 |
| ESPN--33 s2(3682 → 3682 ✗) | **139 tok**(search input 子树) | **0.10** | 0.35 / 0.30 | 同一 chunk |
| Google Map--9 s4(none → none ✗,1023 步未出动作) | **109 tok**(search combobox) | **0.09**(rank 120/153) | 0.15 / 0.17 | — |
| Huggingface--22 s14(12262 → 12262 ✗) | **142 tok**(search input) | **0.08**(rank 118/266) | 0.29 / 0.20 | 0.28 |
| Cambridge--26 s8(991 → invalid ✗) | **219 tok**(search input) | 0.20 | 0.66 / 0.40 | — |
| ESPN--33 s8(15436 → 19502 ✗) | 178 tok(nav) | 0.20 | 0.54 / 0.16 | 0.51 |
| ArXiv--41 s13(2819 → 2819 ✗) | 50 / 23 tok | 0.46 / 0.41 | 0.52 / 0.14 | 0.90(Close modal) |
| ArXiv--41 s10(none → none ✗,1023 步) | 28 tok | 0.65 | 0.29 / 0.34 | — |
| Allrecipes--25 s14(18724 → 16694 ✗) | 27 tok | 0.87 | 0.79 / 0.53 | 0.87 |
| level1-6 s2(1354 → **1417 ✓**) | 136 tok | 0.50 | 0.33 / 0.34 | — |
| 真赢:BBC s4 ✓ / Coursera s12 ✓ / ESPN s11 ✓ | 27 / 41 / 178 tok | 0.41 / 0.82 / 0.56 | — | — |

**机制 ①:大子树里的 needle 被饿死。** 5 个错误的 ref 是 search input / combobox,它们在 WebArena 风格的树里位于 109–219 token 的 `<form>`/`<div>` 子树中(输入框 + 自动补全列表 + 按钮)。mixmax_wn 对大 chunk 有两重惩罚——centroid 被子树里无关的 nav/链接稀释,宽度项又乘了 `(16/L)^0.25`(L=139 时 0.58)——于是这些 chunk 的排名掉到 93–120 位(共 135–266 个),2048 预算下几乎从不入选(0.01–0.20)。fixed16 的 16-token 页单独打分,同一 needle 的页命中 0.15–0.66,高 3–10 倍。小 chunk(23–41 tok)的 needle 命中 0.41–0.87,没有这个问题。

**机制 ②:system prompt 覆盖塌缩。** 实际选择里 system prompt(5.3–5.5k tok)只覆盖 **7–14%** 的 token(B4096 时 36–62%),两个 1023 步仍未产生动作的 "none" 案例(ArXiv s10、Google Map s4)与此一致——模型失去输出格式/流程约束后空转。fixed16-Quest 在离线分析里给 system prompt 27% 的 token、捕获 52% 的 mass(tree 34%)。

两个机制在固定预算下**互相冲突**:给 system prompt 保底(Stage D 的 sys-floor)会从 DOM 那边抽预算,加剧机制 ①;拆大 chunk(`TSA_CHUNK_MAX`)又会让碎片抢小 needle 的预算。需要的是对大 chunk 的"子块打分 + 部分读取"(chunk 按其最好的 16-token 子块打分,入选后只读取最好的几页),这是下一轮离线候选(`cand_sweep2.py`,在这 23 个失败/成功 step 的真实 q 流上评估,不再用旧 dump 的 needle-hit 做代理)。

## 7. 规则表:什么时候选对 / 选错

| 场景 | envelope(quest/mixmax) | legacy 数学 | centroid 家族 |
|---|---|---|---|
| 元素埋在中/大异质 DOM 子树(**最常见**) | ✅ 宽度项捞出(91% 分来自 \|q\|·w) | ❌ 退化 centroid + head 平均变负 | ❌ dilution |
| 元素独占小同质 chunk(cookie/按钮) | ❌ width bias 埋掉 | ⚠️ 靠幅值偶然赢 | ✅ 均值去噪 |
| 指令 / recent 等共识内容 | ⚠️ mass 低,但有 always-include 保底 | ✅ | ✅ |
| 大块 junk chunk(导航 / DOM slab) | ❌ 系统性高估(corr 尺寸 .73–.79) | ✅ | ✅ |
| 检索信号仅在 1–2 个 head | ✅(粗 geometry)head-MAX 保留 | ❌ 平均稀释 | ❌ |
| 细 page(16 tok)× 数百页 | ⚠️ head-MAX 假阳性增多 | ✅ | ✅ centroid≈元素,softmax 反而有害 |
| context >16k、chunk 数 ≫ budget | 全员欠覆盖 —— budget 须随 context 伸缩 | | |

---

### 7.1 追加规则(Stage C/D,2026-08-23)

| 现象 | 条件 | 机制 | 对策 |
|---|---|---|---|
| 任何 faithful 打分与 dense 无差别 | B≥4096,prompt 6–23k | 预算够覆盖 system prompt + DOM 的高 mass 区,selector 不是瓶颈;差异是 decode 随机性 | 用 fidelity(=dense 选择)+ 配对 sign test 判读,不要解读 <5 pt 的 agree 差 |
| tree 的 "真损失" 同配置重跑即变对 | B4096 | batch 组成 / 数值非确定性翻转 argmax | 结论只认可复现的 step;discordant step 先重跑再归因 |
| search-input / combobox 永不入选 | B≤2048,tree_min16,needle 在 109–219 token 子树 | centroid 稀释 + `(16/L)^α` 压低大 chunk;head-mean 抹掉少数 head 的信号 | fixed16 + Block 式 head-max 打分;tree 上的 Block 式/子块打分只能接近不能超过 |
| none / 1023 步空转 | B≤2048,tree | system prompt 覆盖塌缩到 7–14%,输出格式/流程约束丢失 | sys-floor(结构钩子)——online 验证中;离线会降低 needle |
| 离线 needle-hit 预测方向相反 | 旧 dump(非失败富集、q 流来自别的配置) | 代理指标的样本与 q 流与真实失败不匹配 | 只在失败富集的真实 q 流 dump 上评估候选 |

## 8. 对 TSA 的修改清单(按优先级)

1. **修 CUDA selector 的 KV 布局**(§3.5,一行 view 或一个 stride 参数),然后**重跑所有 legacy 基线**——这是 TSA 精度问题的根因,也是之前所有评测结论的前提。
2. **提交容器工作树里的改动**(目前只存在于 spark00 容器,有 .bak 备份):fallback 布局修复(§3.2)、mixmax / legacy_py 实现、TSA_DUMP_DIR 仪表、`test_cuda_layout.py` 作为回归测试。
3. 打分函数:修复布局后 legacy 数学在 B4096 与忠实打分打平;机理上 **mixmax** 仍是最优(needle hit .895、与真实 mass 相关 .890)且与现有 kernel 同构(两个 GEMM + relu 拆分),建议作为默认。
4. budget accounting:若转 token accounting,用 α=0.25 的宽度归一化(§5.5),需 selector 支持 token 贪心选择。
5. budget 随 context 伸缩(Amazon--2 教训)。
6. 用修复后的路径重跑在线评测(此前在线数据:legacy 侧受 §3 bug 影响,vortex 侧受 fallback bug 影响,均作废);在线指标被短 context 步骤稀释,样本量要够。

---


## 8.5 设计空间与相关工作(Stage B/C 的探索框架)

目标:web agent 的 HTML 页面按 DOM 树切成变长 chunk 的 sparse attention,在 **同等 KV 读取量**下 accuracy 与 efficiency 都优于 Quest / BlockSparse 及已有方法。探索维度(已测 ✔ / 进行中 ⏳ / 待做 ○):

| 维度 | 选项 | 状态 / 当前结论 |
|---|---|---|
| 打分函数 | envelope corner-max(legacy)/ 逐维 max(quest, mixmax)/ 宽度归一化 wn(α)/ 子块 envelope(sub16)/ 代表 key(rep)/ centroid(+softmax)/ rank 融合 | ✔ 离线 n=15:chunk-count 口径 mixmax 最优(.885);token 口径 **mixmax_wn025** 最优(.802,超过 fixed16 block .764 / quest .646);sub16 次之 |
| head 聚合 | 组平均 q̄ + head-MEAN / head-MAX / per-head z-score-max / 逐 q-head + MAX(Quest) | ✔ 二阶效应;tree 上 head-MEAN 略优,fixed16 细页上 head-MAX 引入噪声 |
| chunk 几何 | DOM 树原样 / 合并 tiny chunk(min 8/16/32/64)/ 切分大 chunk(max 64/128)/ fixed16 | ✔ 切分有害(needle ≤.72);合并 tiny chunk 到 ≥16 tok 精度不变、被打分单元减半;⏳ min/max 网格扫描中 |
| budget 记账 | top-k chunk(TSA 现状)/ token 贪心 / page 数 | ✔ 三者在 tree 上差异巨大(§5.3),对速度诚实的是 page 数 × page_size |
| page size(内存/attention 单元) | 16 / 32 / 64 / 128 | ⏳ 离线扫描 KV 读取量;Quest/LServe 的经验:page 越大、每页 min/max 越松,精度越差,但 page 越小 page-table 与 kernel 开销越高 |
| 选择频率 | 每层每步(现状)/ 跨 L 层复用 / 每 N 个 decode step 复用 | ○ LServe 报告复用间隔 ≤8 层无显著损失、selector 开销 ↓4×;本仓库已有 layer_chunk_similarity_study 可直接量化 |
| always-include 策略 | sink 4 页 + recent 128 tok(现状)/ 结构感知(任务块、当前 URL、上一条 agent 消息) | ○ 树结构可以定位这些角色 |
| kernel 协同设计 | fallback(PyTorch,现用于实验)/ CUDA selector(需先修 §3 布局 bug)/ mixmax_wn 与 sub16 的 fused kernel / 合并 chunk 减少打分单元 / 子块 envelope 的内存布局 | ○ Stage C:在修复后的 CUDA selector 上实现赢家;sub16 的元数据量 = fixed16 quest 的量 |
| 规模 | 15-step dumps(机理)→ strat20(132 idx 步)→ half1(190)→ full100(369) | ⏳ Stage B 在 strat20 上 end-to-end |

**相关工作与本设计的对应关系**:

- **Quest**(Tang et al. 2024):固定 16-token page,每页逐维 min/max,`Σ_d max(q_d M_d, q_d m_d)` 上界,top-K 页——本报告的 envelope 家族基线;LServe 指出 "Quest … fails when page sizes increase"。
- **LServe**(Yang et al., MLSys 2025):**分层 paging**——物理页由 g 个逻辑页组成,逻辑页各存 min/max,"the importance of each physical page is determined by the max-reduction over the importance scores of its corresponding logical pages"——这正是本报告 `sub16_*` 的结构(tree chunk = 物理页,16-token 子块 = 逻辑页);另有**跨层复用 page selector**("no significant performance degradation until the reuse interval exceeds 8",开销 ↓4×)——我们尚未利用的效率杠杆。
- **ArkVale**(NeurIPS 2024):每页的 key 做 **bounding-volume digest**(包围体,= envelope 的推广)用于 recall/evict。
- **ShadowKV**(ICML 2025):mean-pooled key 做页选择(= centroid 家族)。
- **PBS-Attn**(ICML 2026,"Sparser Block-Sparse Attention via Token Permutation"):固定 block 的问题是"important key tokens … scattered across numerous other blocks",用 token 置换让 block 内更同质——**DOM 树 chunk 就是一种语义置换/分组**,本报告的 tiny-chunk 合并是它的离线版本。
- **Louver**(2026,"Sparse Attention as a Range Searching Problem"):把稀疏 attention 建模为 halfspace range searching,索引结构保证"zero false negatives with respect to a specified threshold"——提示可以给 envelope 上界加 recall 保证的框架。
- **Vortex**(Infini-AI-Lab,2026):可编程稀疏 attention 框架,page 为单位;本报告的 quest/block 忠实实现以其 `flow/algorithms.py` 为参考。
- 其它:MInference / FlexPrefill / XAttention / SeerAttention(prefill 侧模式稀疏,不在本报告范围);RetrievalAttention / MagicPIG / PQCache / ShadowKV(CPU 侧向量检索);DuoAttention(head 级 retrieval/streaming 划分——与 §5.2 的 head 异质性观察相关)。

**本设计与上述工作的差异点**:① 选择单元是 **DOM 结构对齐的变长 chunk**(语义置换的先验版),其余方法都是固定页/块;② 在变长 chunk 上 envelope 的宽度补贴是特有问题,`w·(16/L)^α` 的归一化是针对它的修正;③ 以"page 数 × page_size"的真实 KV 读取量做预算口径,把 chunk 几何、page size、打分、kernel 一起放进同一张 accuracy–KV-read 曲线上比较。


## 9. 运行中的实验与预测

**已落地**:`lpy_B4096v2`(部分,n=75 配对:legacy 数学 54.7% vs CUDA 路径 22.7%,判决已定,停止以让出 GPU);`vxq_B2048` = 47.9%(见 §4.3)。`bt_B4096v2`、`lpy_tree_B4096` 暂缓(优先级让给新方法)。

**Stage B(进行中,strat20 测试床 = 20 任务 / 15 站点 / 209 步 / 132 个 index 步)**。基线在同一 132 步上的 agree:dense 43.2% · fixed16-Block 45.5% · fixed16-Quest 44.7% · fixed16-mixmax 43.2% · fixed16-Quest@B2048 41.7% · tree-Quest 39.4% · tree-mixmax 39.4% · legacy-CUDA 22.0%。四条腿(各 ~1.5 h,自动串行):

| 腿 | 配置 | 对标 | 预测 |
|---|---|---|---|
| `sB_m16_wn25_B4096_p16` | **tree_min16 + mixmax_wn(α=.25) + 4096-token 预算 + page16**(主候选) | fixed16-Block 45.5% / Quest 44.7%(同等 KV 读取) | 机理 needle .797 vs .764/.646 → 预测 ≥46% |
| `sB_tree_wn25_B4096_p16` | 同上但不合并 tiny chunk | 消融:合并的影响 | ≈ 主候选,打分单元 ×2 |
| `sB_m16_mixmax_k64_p64` | tree_min16 + mixmax + top-64 chunk(chunk-count 口径) | tree-mixmax 39.4% | 合并单独的效果;读取 ~7k token,不与 fixed16 同口径 |
| `sB_m16_wn25_B2048_p16` | 主候选的 2048-token 版 | fixed16-Quest@B2048 41.7% | 低预算区 tree 反超假说 |

之后自动续跑 `vxt_B2048`(tree k32 Quest,half1 全量)。

**Stage B 第一腿结果(2026-08-22,strat20 的 132 个 index 步,0 错误,4.9 h)**:

| 配置(同一 132 步) | valid | agree | agree% |
|---|---|---|---|
| fixed16-Block | 97 | 60 | 45.5% |
| fixed16-Quest | 99 | 59 | 44.7% |
| **tree_min16 + mixmax_wn(α=.25) + 4096-token + page16(新)** | **104** | **58** | **43.9%** |
| tree(不合并)+ mixmax_wn + 4096-token + page16(消融) | 104 | 54 | 40.9% |
| tree_min16 + mixmax_wn + **2048**-token + page16(第三腿) | 92 | 48 | 36.4% |
| dense | 99 | 57 | 43.2% |
| fixed16-mixmax | 100 | 57 | 43.2% |
| fixed16-Quest @B2048 | 100 | 55 | 41.7% |
| tree-Quest / tree-mixmax(旧 tree 最好) | 99 / 100 | 52 / 52 | 39.4% |
| legacy-CUDA tree | 87 | 29 | 22.0% |

解读:① 新方法把 tree 几何此前落后 fixed16 的 6-8 步差距补上了(52 → 58),在同等 KV 读取量下进入 fixed16 的顶部簇(与 Block 差 2 步、与 Quest 差 1 步,均在 n=132 的噪声内),valid 最高、invalid 为 0;② 被打分的单元约为 fixed16 的 1/4(合并后 ~120 个 chunk vs 541 页);③ 还**不能**宣称显著超过 Block——分站点看,Huggingface(6 vs 4)、ArXiv(9 vs 8)占优,Apple / Coursera / ESPN / GitHub / Wolfram 各少 1 步。接下来:不合并消融(第二腿)、B2048 版(第三腿,对标 Quest@B2048 的 55),然后把赢家扩到 half1 全量 190 步以获得统计功效。

**第二腿(不合并消融)结果**:54/132(40.9%),比合并版少 4 步(配对:合并独赢 8 步、不合并独赢 4 步)。离线 needle 指标曾显示两者精度相同,end-to-end 却差 4 步——机理推测:宽度归一化因子 `(16/L)^α` 在 L<16 时大于 1(L=4 时 1.41),未合并的 1–8 token 碎片被抬分、挤占 token 预算,选中内容缺乏上下文。**结论:把 tiny chunk 合并到 ≥16 token 是方法的必要组成部分,而不只是省算力的开关**;离线指标应补一项"选中 chunk 的碎片比例"。
实现:`tree_sparse_selector.py` 新增 `TSA_SCORING_IMPL=mixmax_wn`(`TSA_WN_ALPHA`)、`TSA_CHUNK_MIN/MAX`、`TSA_BUDGET_TOKENS`,与离线参考定义对拍(max|Δ| ≈ 4e-6、0 rank 翻转、几何逐一相同、预算 = 贪心前缀)。

**第三腿(B2048)进行中的方向性读数与两个离线补充实验(2026-08-22 晚)**:48 步配对上新方法@B2048 18/48,fixed16-Quest@B2048 22/48(同 48 步上新方法@B4096 26、Block 26、dense 24)——"低预算区 tree 反超"假说目前看是反的。两个离线实验解释了原因:① LServe 式分层选择(16-token 子块 + β×父 chunk 分数)needle 不高于纯 chunk 级选择(B2048:.561 vs ≤.545),且子块跨页导致 KV 读取更高;② **紧预算下 fixed16 的 mass recall 反而更高**(B2048:.853 vs .830)——细粒度把预算分散到更多区域,模型在紧预算下更需要广度;混合分配(f 份给 tree chunk、其余给 fixed16 页)是线性折中、无占优点。结论:**tree 整块选择的适用区是 B≥4096(needle 相同、打分单元 1/4),≤2048 是 fixed16 细粒度的主场**。计划相应调整:第三腿跑完后把 B4096 主候选扩到 half1 全量 190 个 index 步(`sC_m16_wn25_B4096_p16`,~7 h)以取得统计功效,再跑 vxt_B2048 基线。

**第三腿(B2048)最终结果**:48/132(36.4%),valid 92,none 40(全场最高)vs fixed16-Quest@B2048 55(41.7%);配对:Quest 独赢 13 步、新方法独赢 6 步。**紧预算下 tree 整块选择明确输给 fixed16 细粒度**,且 none 偏高说明模型在紧预算下"失去方向"——与离线的 mass recall 差(.830 vs .853)一致:2048 token 花在 ~40 个整块上,覆盖的区域太少。

**Stage B 小结(strat20,132 个 index 步)**:

| 口径 | 最强 fixed16 基线 | tree 新方法 | 差 | 结论 |
|---|---|---|---|---|
| B4096(≈4.4k KV 读取) | Block 60 / Quest 59 | **58**(合并版) | −1~−2(噪声内) | 追平,打分单元 1/4,valid 最高 |
| B4096 不合并消融 | — | 54 | −4 vs 合并 | 合并 tiny chunk 是必要组件 |
| B2048 | Quest 55 | 48 | −7 | 紧预算 fixed16 胜(广度) |

下一步:主候选(B4096 合并版)扩到 half1 全量 190 个 index 步(`sC_m16_wn25_B4096_p16`,运行中)取统计功效;紧预算短板的改进方向(离线先验证):按 DOM 区域给预算配额保证广度、或 chunk 内按子块取 top 而非整块。

**紧预算短板的第三个离线尝试——按窗口配额强制广度——同样失败**(B2048 needle .24–.47、mass .80–.82,均低于纯 chunk 选择的 .561 / .830):把预算硬摊到各窗口只会浪费在无关区域。三次离线尝试(子块分层、tree+fixed16 混合、窗口配额)都没能让 tree 在 ≤2048 同时保住 needle 和 mass,因此当前的诚实结论是**按预算自适应选粒度**:B≥4096 用 DOM 树整块选择(追平 fixed16、打分单元 1/4),≤2048 用 fixed16 细粒度。第四个离线尝试——chunk 内优先保留首个 16-token 子块(`[idx]<tag` 前缀所在)——在相同 token 预算下 B2048 needle 升到 .618(parent-only .561、fixed16 .439)、mass 持平,但子块不对齐 page,实际 KV 读取多 26%(2957 vs 2344),按同等 KV 折算收益基本消失;判定为弱信号,不上 GPU。

**Stage C 扩展腿结果(2026-08-23,half1 全量 190 个 index 步,50 任务 / 15 站点,0 错误,6.4 h)**——主候选 `sC_m16_wn25_B4096_p16` 与全部基线在同一 190 步上配对(`analyze_stageC.py`):

| B=4096 tok,page16(同一 190 步) | valid | agree | agree% | 与 dense 选同一 action | 真损失 / 侥幸 | vs Block 独赢/独输 | sign p | 差值 bootstrap 95% |
|---|---|---|---|---|---|---|---|---|
| dense(参考) | 154 | 97 | 51.1% | 100% | — | 6 / 8 | 0.79 | [−9, +5] |
| fixed16-Block | 151 | **99** | 52.1% | 76% | 6 / 8 | — | — | — |
| fixed16-Quest | 151 | 98 | 51.6% | 77% | 9 / 10 | 7 / 8 | 1.00 | [−9, +7] |
| fixed16-mixmax | 155 | 96 | 50.5% | 77% | 9 / 8 | 8 / 11 | 0.65 | [−11, +6] |
| **tree_min16 + mixmax_wn(α=.25) + 4096-token(新)** | **156** | 96 | 50.5% | **79%** | 10 / 9 | 8 / 11 | 0.65 | [−11, +6] |
| tree-mixmax k64(旧 tree 最好) | 157 | 91 | 47.9% | 80% | 10 / 4 | 4 / 12 | 0.08 | [−16, 0] |
| tree-Quest k64 | 151 | 89 | 46.8% | 76% | 12 / 4 | 3 / 13 | **0.02** | [−18, −3] |
| legacy-CUDA tree(布局 bug) | 134 | 55 | 28.9% | 47% | 45 / 3 | 5 / 49 | <0.001 | [−57, −31] |

("真损失" = dense 对、该方法错;"侥幸" = dense 错、该方法对;sign test 与 bootstrap 均为与 fixed16-Block 的逐步配对。)

读法:
1. **B4096 是"透明区"**:dense 本身对 Block 也是 6 赢 8 输,所有 faithful 的 sparse 方法(fidelity 76–80%)与 dense 落在同一噪声带,agree 指标在 n=190 下分不出 <5 pt 的差异(bootstrap 宽度 ±8 步)。新方法与 Block/Quest/dense 的差异均不显著。
2. **在 tree 几何内部,新方法是唯一追平 fixed16 的配置**:tree-Quest k64 显著输给 Block(p=0.02),tree-mixmax 边缘(p=0.08);新方法 p=0.65。即"合并 tiny chunk + 宽度归一化 + token 预算"把 tree 几何从显著落后拉到统计上不可区分,同时被打分的单元约为 fixed16 的 1/4、valid 最高(156)。
3. **不能宣称超过 fixed16 基线**。Block 赢的 11 步里 5 步是真损失,且全是相邻 index 错误(492 vs 493、31448 vs 31450、8117 → 81117),其余 6 步 dense 也错(侥幸);分站点新方法在 Huggingface 7 vs 4、Amazon 8 vs 7 占优,Allrecipes 9 vs 12、ESPN 7 vs 9 落后。n=150 时一度落后 6 步,最后 40 步追回——再次说明这是噪声带内的波动。
4. 对照 B2048(strat20 n=132):tree 48 vs fixed16-Quest 55,fidelity 57% vs 66%,9 个真损失 vs 4 个真赢,损失形态是选到远处元素或输出非 index 动作——**紧预算的劣势是真的**,而离线 needle-hit 在 B2048 上 tree 反而更高(.561 vs .439)、mass recall 略低(.830 vs .853)、page16 粒度实际多读 14% KV(2344 vs 2057)。结论:needle-hit 不是正确的代理指标,之前四个离线方案的"失败"判断需要用真实失败 step 的 dump 重新校准。

**Stage C 后续(运行中)**:对 B4096 的 10 个 discordant step 与 B2048 的 13 个真损失/真赢 step 各跑一次 dump 腿(相应配置,batch 1,保存全层 K / 每步 q / 每层 top-k 选择),用 `case_discord10.py` 回答:ref 元素所在 chunk 是否真的被选、同一 query 下 fixed16 Block/Quest 是否覆盖、ref chunk 的 attention-mass rank、模型选错的元素是否被覆盖;另有 `region_sweep.py`(CPU,15 个旧 dump)测 B2048 预算在 system prompt / task / DOM / tail 各区域的分配与 mass 捕获率。之后补 `vxt_B2048` 基线。

**Stage D(2026-08-23,紧预算机制实验)**。依据 §6.10 的两个机制,在 23 个新 dump(13 个 B2048 失败/成功 step + 10 个 B4096 discordant step,真实 q 流,已知 ref)上做离线候选 sweep(`cand_sweep2.py`,B=2048,每 step 先取均值):

| 变体(单元 / 打分 / sys-floor) | needle 命中 | sys 覆盖 | DOM 覆盖 | mass | KV 读 @16 |
|---|---|---|---|---|---|
| **fixed16 / Block(softmax-centroid,head-max)top-128** | **0.537** | 0.158 | 0.307 | 0.531 | 2048 |
| tree_min16 / mixmax_wn(当前方法) | 0.479 | 0.105 | 0.419 | 0.799 | 2341 |
| tree_max128 / wn | 0.436 | 0.132 | 0.370 | 0.806 | 2339 |
| tree_min16 / wn / floor .3 | 0.434 | 0.147 | 0.361 | 0.791 | 2348 |
| tree_max64 / wn | 0.392 | 0.147 | 0.322 | 0.812 | 2402 |
| tree_min16 / 子块 wn 取 max | 0.356 | 0.182 | 0.270 | 0.780 | 2285 |
| fixed16 / Quest top-128 | 0.352 | 0.196 | 0.216 | 0.771 | 2048 |
| tree_min16 / wn / floor .5 | 0.330 | 0.205 | 0.273 | 0.777 | 2335 |
| tree_min16 / 子块 Quest 取 max | 0.274 | 0.233 | 0.168 | 0.698 | 2286 |

三个判断:① 在失败富集的 step 上,**fixed16-Block 的 needle 命中最高**(0.537),而它的 mass 捕获最低(0.531)——Block 的 softmax 归一化 + head-max 选的是"某个 head 最想看的页",对 search-input 这类只被少数 head 关注的 needle 更敏感;mixmax_wn 的 head-mean 把它们平均掉了。B2048 的 online 基线此前只跑了 Quest(55),**Block@B2048 是缺失的、更强的基线**,已加入队列。② 所有"修复大 chunk"的尝试(拆 max64/128、子块取 max、trim64)在离线上都降低 needle 命中:拆分/子块让碎片与小 needle 抢预算,而子块 wn 本身就不命中这些页(是 head 聚合的问题,不是粒度)。③ sys-floor 单调降低 needle(.479 → .434 → .330),换来 sys 覆盖 .105 → .205;它只可能通过减少 none/空转起作用,online 的 `sD_m16_wn25_B2048_sf5` 正在跑,`sf3` 腿换成 Block@B2048 基线(`sD_blk16_B2048`)。下一批离线候选:tree chunk 上的 Block 式打分(softmax-centroid、head-max)、子块 Block 取 max、head-max 版 wn(`cand_sweep2b.py`,运行中)。

**第二批离线候选(`cand_sweep2b.py`,同 23 个 dump,B=2048)**——tree chunk 上的 Block 式打分:

| 变体 | needle | sys 覆盖 | DOM 覆盖 | mass | KV 读 @16 |
|---|---|---|---|---|---|
| fixed16 / Block top-128(基线) | **0.537** | 0.158 | 0.307 | 0.531 | 2048 |
| tree_min16 / 子块 Block 取 max / trim64(大 chunk 只读最好 4 页) | 0.515 | 0.139 | 0.309 | 0.501 | 2073 |
| tree_min16 / chunk 级 Block(centroid softmax,head-max)/ trim64 | 0.506 | 0.132 | **0.355** | **0.580** | 2174 |
| tree_max64 / 子块 Block | 0.502 | 0.161 | 0.292 | 0.461 | 2335 |
| tree_min16 / chunk 级 Block | 0.484 | 0.163 | 0.312 | 0.514 | 2325 |
| tree_min16 / mixmax_wn(当前) | 0.479 | 0.105 | 0.419 | **0.799** | 2341 |
| tree_min16 / head-max 版 wn | 0.477 | 0.131 | 0.360 | 0.789 | 2358 |

结论:把 Block 的 head-max + softmax 归一化搬到 tree 单元上,needle 能从 .479 提到 .506–.515,**接近但不超过 fixed16-Block 的 .537**;代价是 mass 从 .80 掉到 .50–.58(Block 式打分不选 sink / system prompt 的高 mass 块)。没有任何 tree 变体在 needle 上赢过 fixed16-Block。另一个反直觉的事实:在这 13 个 B2048 失败 step 上,tree-wn 的 needle 命中反而高于 online 赢了的 Quest(8/13 step 更高,例如 Allrecipes--25 s14:.90 vs .54),说明**紧预算下决定 online 结果的不只是 needle 在不在,还有上下文的组成**(system prompt 覆盖 7–14% vs Quest 的 ~20%)。这正是 `sD_m16_wn25_B2048_sf5` 在测的假设;Block@B2048 基线(`sD_blk16_B2048`)随后。

**B2048 基线补齐(2026-08-23,`sD_blk16_B2048`,strat20 的 132 个 index 步,0 错误,4 h;`analyze_stageD.py` 配对)**:

| B=2048(同一 132 步) | valid | none | agree | fidelity(=dense) | 真损失 / 侥幸 | vs Quest@B2048 独赢/独输 | sign p | bootstrap 95% |
|---|---|---|---|---|---|---|---|---|
| dense(参考) | 99 | 33 | 57 | 100% | — | 10 / 8 | 0.81 | [−6, +11] |
| **fixed16-Block@B2048** | **106** | **25** | 55 | 63% | 10 / 8 | 10 / 10 | 1.00 | [−8, +9] |
| fixed16-Quest@B2048 | 100 | 31 | 55 | 66% | 10 / 8 | — | — | — |
| tree_min16 + mixmax_wn@B2048 | 92 | 40 | 48 | 57% | 15 / 6 | 6 / 13 | 0.17 | [−16, +2] |
| (对照)fixed16-Block@B4096 | 97 | 34 | 60 | 71% | 4 / 7 | 12 / 7 | 0.36 | |
| (对照)tree_min16 + wn@B4096 | 104 | 28 | 58 | 70% | 7 / 8 | 11 / 8 | 0.65 | |

读法:紧预算下 fixed16 的两种打分 online 完全打平(55 / 55;Block 的 valid 最高、none 最少),tree 落后 7 步(真损失 15 vs 10),与 §6.10 的机制分析一致;fidelity 57% vs 63–66% 说明差距是真实的而非噪声带波动(B4096 的 fidelity 70–71%)。`sD_m16_wn25_B2048_sf5`(sys-floor 0.5)在 Block 腿之后重跑(首次因等待循环的竞态被误杀,见 §10 的教训)。

**sys-floor 腿终表(`sD_m16_wn25_B2048_sf5`,132 步,0 错误)**:

| B=2048(同一 132 步) | valid | none | agree | fidelity | 真损失 / 侥幸 |
|---|---|---|---|---|---|
| dense | 99 | 33 | 57 | 100% | — |
| fixed16-Block / Quest | 106 / 100 | 25 / 31 | 55 / 55 | 63% / 66% | 10 / 8 |
| tree-wn(无 floor) | 92 | 40 | 48 | 57% | 15 / 6 |
| **tree-wn + sys-floor 0.5** | 105 | 26 | **50** | 49% | 19 / 12 |

判读:floor 把"空转"机制修掉了(none 40 → 26,valid 92 → 105,都回到 fixed16 水平),net +2 步(48 → 50),**但仍落后 fixed16 5 步**,且 fidelity 掉到 49%、真损失 19、侥幸 12——把预算的一半锁给 system prompt 后,DOM 侧的选择更像掷骰子(离线 needle .479 → .330 的预言兑现)。结论:§6.10 的两个机制彼此独立且都真实——sys-floor 只修机制 ②,机制 ①(大子树 needle 被饿死)没有 tree 内的解(§9 第二批离线候选:一切"修大 chunk"的打分都 ≤ fixed16-Block)。**紧预算的最终结论:fixed16(Block=Quest=55)> tree 任何已试变体(≤50);tree 的适用区从 B≥4096 起算**(B3072 交叉点腿运行中)。

**Selection-cost 微基准(`bench_select.py`,GPU 空载,真实 prompt 的 token 流 + 合成 K/q,warmup 后 100 次 select × 48 层取均值)**:

| 路径 | 单元 | T=6.2k(单元数) | T=23k(单元数) | select µs/层·次 | 折合 ms/decode-token(48 层) | prefill 统计 ms/层 |
|---|---|---|---|---|---|---|
| **CUDA kernel(修复后)** | tree_min16 k64 | 56 | 192 | 50–76 | **2.4–3.7** | 0.17–0.27 |
| CUDA kernel | fixed16 top-256 | 386 | 1447 | 74–120 | 3.5–**5.8** | 0.19–0.35 |
| PyTorch fallback(mixmax_wn + token 预算) | tree_min16 B4096 | 56 | 192 | 1475–1613 | 71–77 | 3.7–8.1 |
| PyTorch fallback(vortex Quest/Block) | fixed16 top-256 | 386 | 1447 | 873–1613 | 42–77 | 16–94 |

三个结论:① **CUDA 路径上,单元数 1/4–1/7 带来 select 每次调用 1.3–1.6× 的加速**(76 vs 120 µs @T=23k;调用有固定开销,加速比小于单元数比),折合每 decode token 省 ~2 ms(48 层);② **PyTorch fallback 的选择开销(42–77 ms/token)与整个模型 forward 同量级**——这就是本轮所有腿 110–145 s/step 的主因,也是当前所有 end-to-end 数字的共同税负,对各配置一视同仁(配对比较不受影响);③ 工程上收益最大的一步是把 mixmax_wn + token 预算(+ sys mask)移植进 CUDA selector(现在 CUDA 路径只支持 chunk-count top-k,所以表中 CUDA tree 行读的 KV 是 6.2k–7.9k token,不是 4096 预算)。

**B3072 交叉点腿终值(`sE_m16_wn25_B3072_p16`,132 步,0 错误)**:tree@B3072 **agree 57/132(43.2%)= dense(57)**,fidelity 66%、真损失 8 / 侥幸 8、none 28、valid 101——完全恢复到 B4096 形态(58,fidelity 70%),显著高于 tree@B2048 的 48,也高于两个 fixed16@B2048 基线(55)。**tree 的塌缩边界在 2048 → 3072 之间**,与机制吻合:3072 刚好同时容纳 system prompt 的高 mass 段(top-10 chunk ≈ 2.4k tok 的一半多)与 DOM needle 区。`sE_blk16_B3072` 终值:**53/132(fidelity 72%,none 28)**——tree@B3072 57 vs Block@B3072 53,直接配对 tree 赢 10 / Block 赢 6(sign p≈0.45,不显著),两者都在 dense(57)的噪声带内。**B3072 交叉点结论:tree 在 3072 已完全恢复且不落后于 fixed16;"按预算自适应粒度"的切换阈值定为 ~3k token(约为 system prompt 高 mass 段 + DOM needle 区的联合覆盖需求);B<3k 用 fixed16(55 vs 48–50),B≥3k tree 与 fixed16 等价、打分单元 1/4。**

**复用机会量化(`reuse_analysis.py`,真实 dump,CPU,2026-08-24)**:

- **请求内选择稳定性**(23 个 dump × 48 层的 topk.jsonl):相邻 decode 步选中 chunk 集的 Jaccard = **0.81**,隔 8 步 0.71、隔 32 步 0.67;逐层看 lag-1 在 0.70–0.93(最不稳定的是 22–30 层)。含义:LServe 式"每 k 个 token 重选一次"在 k=4–8 时选择集与新鲜选择重叠 ~71–78%,选择开销 ÷k(与 CUDA 移植叠加后 select 可到 <1 ms/token);**陈旧选择对 agree 的影响未测**,是移植后的首个验证项。
- **跨步前缀复用**(同一任务相邻 dump 步,8 对,去掉 2 对重复步):后一步 token 流的 **35–79%(均值 ~59%)是与前一步逐字节相同的前缀**(system prompt ~5.4k + 未变的任务/历史头部),该前缀内 chunk 的 K / envelope / centroid 位置不变、可直接缓存复用。诚实注记:这本质是标准 prefix caching(fixed16 页在相同前缀里同样不变),**不是 tree 特有的优势**;tree 特有的"页面中段未变子树复用"被 RoPE 的位置依赖挡住(位置移动 → K 变),需要 CacheBlend 类的位置修复才谈得上。

**时间复用验证(`sF_m16_wn25_B4096_rs8`,132 步,0 错误)**:tree_min16+wn@B4096 改为**每 8 个 decode 步重选一次**(其间沿用缓存的 chunk 选择;`TSA_RESELECT_K=8`,补丁 `patch_reselect.py`):agree **55 vs fresh-select 58**(dense 57;配对 7/10,p≈0.63,不显著)、valid 105 vs 104、fidelity 81 vs 93(输出确实被陈旧选择改变了,但落在噪声带内)、wall-clock **105 vs 133 s/step**(−21%,conc6 摊薄后的 PyTorch 路径收益)。结论:**LServe 式时间复用在 k=8 下端到端无显著代价**,与选择集 lag-8 Jaccard 0.71 的离线测量一致;叠加 CUDA 移植后 selection 成本可到 ~0.4 ms/token 量级。

**矩阵补全(`vxt_B2048`,half1 全量 332 步 / 190 idx,0 错误,11.4 h)**:tree-Quest k32(chunk-count 口径,实际 KV ≈2k+ token)agree **82/190** vs fixed16-Quest@B2048 91 vs dense 97——half1 规模上复现 strat20 的紧预算结论(tree −9)。至此本轮管线(dump 复盘 ×2、区域/集中度/复用分析、sys-floor、B3072 交叉点、时间复用、selection 微基准、全部基线)**全部完成,0 错误**。

**Stage C(待 Stage B 结果)**:赢家扩到 half1 全量 332 步 → full100;修复 CUDA selector 布局后把 mixmax_wn / sub16 做成 fused kernel;跨层复用 selector(LServe 式)的效率实验;online 重跑。

## 10. Artifact 索引

| 内容 | 路径(Mac,repo 相对) |
|---|---|
| **布局 bug 复现(真实路径,决定性)** | `simulator/runs/sparse3way-20260721/scoring_case_study/test_cuda_layout_realpath.py`(spark00:`/workspace/scoring_case_study/`) |
| 布局 bug 复现(合成 buffer,机理版) | `simulator/runs/sparse3way-20260721/scoring_case_study/test_cuda_layout.py` |
| end-to-end summary/result(全部 tag) | `simulator/runs/sparse3way-20260721/{summary,result}_*.json(l)` |
| 分类别对照表 | `simulator/runs/sparse3way-20260721/per_category_scoring.md` |
| 机理 sweep(n=15 全量)/ α 扫描 | `.../scoring_case_study/analysis/sweep_v3.md`, `sweep_wn.md`, `summary_v3.json` |
| 逐步机理指标 | `.../scoring_case_study/analysis/per_step_metrics*.jsonl`(v3 全量 36 MB 在 spark00) |
| Case studies 完整版 | `.../scoring_case_study/analysis/cases.md`, `cases_centroid.md` |
| dilution 定量数据 | `.../scoring_case_study/analysis/cases_centroid_{data,centered}.json` |
| capture 清单 / 响应 | `.../scoring_case_study/{capture_manifest.json,responses/}` |
| 服务器侧 | spark00 容器 `/workspace/scoring_case_study/`(dumps ~10 GB、chain 日志)、`/workspace/sparse3way/` |
| 代码改动(未提交,容器工作树) | `python/tree_sparse_selector.py`(fallback 布局修复 + mixmax + legacy_py + dump 仪表;备份 `.bak_predump / .bak_premixmax / .bak_prelegacypy`);**CUDA 路径未修** |

**复现参数**:`serve.py --max-decode-tokens 4096 --max-batch-size 8 --batch-collect-ms 150 --disable-cuda-graph`;tree = `--tree-parse-mode webarena --page-size 64`,fixed16 = `--tree-parse-mode fixed --page-size 16`;replay 客户端 temperature 0、xgrammar `tool_schema.json`;GB10 上必须禁用 CUDA graph(MoE grouped_mm 崩溃)。

### 10.1 2026-08-23 新增 artifact(Stage C/D)

| 路径 | 内容 |
|---|---|
| `runs/sparse3way-20260721/result_sC_m16_wn25_B4096_p16.jsonl` + `summary_*.json` | Stage C 主候选 half1 190 步结果 |
| `runs/sparse3way-20260721/result_sD_blk16_B2048.jsonl` | fixed16-Block@B2048 基线(strat20 132 步) |
| `runs/sparse3way-20260721/result_sD_m16_wn25_B2048_sf5.jsonl` | tree + sys-floor 0.5 @B2048(运行中) |
| `runs/sparse3way-20260721/analyze_stageC.py` / `analyze_stageD.py` | 配对分析:agree / fidelity(=dense)/ 真损失·侥幸 / sign test / bootstrap |
| `runs/sparse3way-20260721/case_dumps/case_discord10_B4096.{md,json}` | §6.9 的 10 个 B4096 discordant step 复盘 |
| `runs/sparse3way-20260721/case_dumps/case_discord2048_B2048.{md,json}` | §6.10 的 13 个 B2048 step 复盘 |
| `runs/sparse3way-20260721/discord2048.json` | B2048 真损失 / 真赢 step 列表 |
| spark00 `/workspace/scoring_case_study/dumps_discord10/`, `dumps_discord2048/`(6.9 G + 9.1 G) | 全层 K、每步 q、每层实际选择(topk.jsonl)、chunks.json(含 user_start) |
| spark00 `case_discord10.py` | dump 复盘分析器(`--budget`,region 分配、needle 实际命中、同 query 离线 Block/Quest 命中、mass rank) |
| spark00 `region_sweep.py`, `sys_concentration.py` | 区域分配 / system prompt mass 集中度(`analysis/region_B2048*.log`, `sys_concentration.log`) |
| spark00 `cand_sweep2.py`, `cand_sweep2b.py` | 紧预算候选 sweep(`analysis/cand_sweep2_B2048.log`, `cand_sweep2b_B2048.log`, `*.json`) |
| spark00 `patch_sysfloor.py` → `tree_sparse_selector.sysfloor.py`(已装入 `python/tree_sparse_selector.py`,备份 `.bak_presysfloor2`) | `TSA_SYS_FLOOR=f`:token 预算下给第一个 user turn 之前的 chunk 保底 f·budget;`chunks.json` 增 `user_start` |
| spark00 `bench_select.py`, `analysis/bench_select.jsonl` | selection-cost 微基准(CUDA / PyTorch 路径,tree_min16 vs fixed16) |
| spark00 `run_dump_leg.sh`, `chain_dump_v2.sh`, `chain_after_{dumps,sD1,blk,sf5}.sh`, `dump10_chain.log` | 链式调度脚本与日志(含一次等待循环竞态误杀 sf5 的记录) |

