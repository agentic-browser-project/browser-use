# mixmax 为什么可以不像 Quest 那样逐 head 计算——它到底是通用改进还是 web-agent 改进

**日期**:2026-08-28 · **模型**:Qwen3-VL-30B-A3B(32 query heads / 4 KV heads,GQA group=8)· **数据**:WebVoyager+GAIA 真实轨迹 offline replay(细节与全部原始数字见 `2026-08-23_scoring_function_study.zh.md`,本文引用其编号)· **代码**:`BrowserSparseAttention` 分支 `shiqihe/mixmax-main`

---

## 0. TL;DR

1. mixmax 是三个成分的组合,**三个成分的"归属"不同**:单元打分公式(逐维 corner max 上界)**完全来自 Quest**;GQA group-mean 的 head 处理**来自 TSA 既有 kernel 结构**(工程适配,顺带省 8× 计算);宽度归一化 `(16/L)^α` 是**本工作里唯一少见的成分**,而它只在变长单元下才有意义。
2. "可以不逐 head 算"不是理论保证,是**分区间的实证结论**:KV 读取 ≥3k token 时,head-mean 与 head-max 端到端不可区分(fixed16 上 mixmax 96 vs Quest 98 vs Block 99 vs dense 97,n=190,全在噪声带);≤2k 时 head-mean 有真实代价(retrieval-head 信号被组内平均稀释,§3)。web agent 的实际工作点在前一个区间,所以拿 8× 计算节省不付准确率。
3. **它不是"web-agent 的打分改进",而是"让 web-agent 的几何可用的打分适配"**:效率收益(打分单元 5–7× 少 → select 快 1.3–1.55×)来自 DOM 树变长 chunk 这个几何;Quest 的上界直接搬到变长 chunk 会因包围盒随长度膨胀而失效(agree 98→89,p=0.02),wn 补的就是这个洞。公式本身通用,几何才是 web-agent 特有的。
4. 后 Quest 的方法几乎全部停留在**固定大小单元**上,所以长度归一化的问题在文献里基本不出现;head 聚合各家选择不一但很少被单独 ablate。我们的贡献是把这两个轴拆开做了控制变量的 A/B,并给出失效机制的 case study。

---

## 1. 先把 mixmax 拆成三个成分

对每个候选单元(chunk/page)存包围盒:逐维 max 向量 M、min 向量 m(每 KV head 一套)。

| 成分 | 公式 | 来源 | 作用 |
|---|---|---|---|
| ① 单元上界 | s = Σ_d max(q_d·M_d, q_d·m_d) = q·(M+m)/2 + Σ_d\|q_d\|·(M−m)_d/2 | **Quest 原样** | 盒内任意 key 与 q 点积的精确上界 |
| ② head 处理 | q̄_h = 组内 8 个 query head 的平均;4 个 KV-head 分数取 mean | **TSA/legacy 既有结构** | 每单元 4 次点积而不是 32 次(8×);与现成 fp8 kernel 同构 |
| ③ 长度归一 | 宽度项 × (16/L_c)^0.25 | **新增** | 消掉"盒子随 chunk 变长必然膨胀"的乐观偏置;变长 chunk 可用的关键 |

对照:Quest = ① + 逐 head 原始 q + head-**max**,固定 16-token page,无 ③。TSA 原 2corner = ② + 把 max 套在**求和外面**(\|Σq̄_d w_d\|,正负抵消 → 宽度项归零 → 退化成 midrange-centroid;`2026-08-23_scoring_function_study.zh.md` §1.1/§5)。所以 mixmax 的准确描述是:**把 Quest 的逐维上界装进 TSA 的 head-mean 结构里,再为变长单元加长度修正**。

## 2. 为什么 head-mean 可以:实证 + 机理

**语义上确实变弱了**:Quest 的分数是"任何一个 head 能取到的点积"的上界(逐 head 上界再 max);mixmax 的分数只是**平均 query** 的上界,对单个 head 不再成立(mean ≤ max 恒成立,单 head 强信号可被组内其余 7 个平均掉)。那为什么敢用?

1. **工作区间的实证**(`2026-08-23_scoring_function_study.zh.md` §4.1/§9 Stage C):B4096、190 步配对,fixed16 几何上 mixmax(head-mean)96 vs Quest(head-max)98 vs Block 99 vs dense 97——配对 sign test 全不显著,dense 自己重跑也在同一带里。**中等稀疏度下,排序尾部的差异根本不进入被选集合**:needle 即使被平均稀释,仍稳稳排进 top-60 chunk / 4096 token 预算内。
2. **GQA 组内相关性**:head-mean 只在**共享同一 KV head 的 8 个 query head 内部**做平均,不跨组。同组 head 对同一 KV 空间打分,相关性天然高——main 分支上团队自己的分析(`results/per_head_recall`、"per-step within-GQA-group agreement/consensus" 系列图、Qwen3-32B head-disjointness)研究的正是这个量;组内一致性越高,mean 相对 max 的信息损失越小。(他们还在 kernel 里做了 mean/adakv/mean_raw 的 head-combine ablation 接口——说明这在他们那边也是活跃问题,不是我们发明的轴。)
3. **计算结构**:TSA 的 fp8 打分 kernel 本来就是"分组平均后的 q 对每 KV head 一次点积";mixmax 塞进去不用改数据通路。Quest 的逐 head 版要 32 份分数 + max reduction(8× 计算,或像 vortex 那样 bmm 到 [K,G,P] 再 amax)。

## 3. 什么时候不行:head-mean 的真实失效区(case study 有讨论)

`2026-08-23_scoring_function_study.zh.md` §6.10 + §9 的候选 sweep 正面回答过这个问题:

- **B2048(紧预算)真实损失的 5/9 是 search-input/combobox**,它们在 109–219 token 的 DOM 子树里,tree+head-mean 打分下实际被选中率 **0.01–0.20**;同一 query 流上 fixed16 页命中 0.15–0.66。
- 换打分不换几何的离线 A/B(23 个失败富集 dump,cand_sweep2/2b):fixed16-**Block(head-max softmax)needle 0.537** > tree head-mean wn 0.479 > fixed16-**Quest(head-max 上界)0.352**;把 Block 式 head-max 搬上 tree 单元最高 0.515,仍不超过 fixed16-Block。结论当时就写了:**"是 head 聚合的问题,不是粒度"**——检索信号集中在少数 head("retrieval heads"现象),mean 把它 ÷8;但也注意 Quest 的 head-max **上界**在细 page 上双重乐观(max over heads × max over corners)反而放大噪声,head-max 不是免费午餐(§6 case 6 对 softmax 的检验同理)。
- 在线终值:B2048 fixed16 Quest=Block=55 > tree 全部变体 ≤50;**B3072 起 tree 恢复**(57 = dense,Block 53)。所以边界清楚:**≥3k 用 head-mean 免费省 8×;<3k 是 head-max 的地盘,而那个区间 fixed16 几何本来也赢**,tree+mixmax 根本不该在那儿用。

## 4. 它到底是不是 web-agent 特有的提升

分解回答:

- **公式(①+②)完全通用**:任何 GQA 模型、任何 workload 都能用,没有一个字和 HTML 有关。它的存在理由一半是工程(TSA kernel 同构),这属于 **TSA-stack 特有**的好处。
- **web-agent 特有的是几何,不是打分**:DOM 树切出的变长 semantic chunk 在真实页面上是 78–400 个单元,fixed16 是 424–1672 个(5.4–7×);select 实测 53.7/73.7 µs vs 74.1/114 µs(1.3–1.55×,kernel 启动开销吃掉了一部分理论比)。而"≥3k 才透明"这个阈值本身也来自 web-agent 的 prompt 结构(system prompt 高 mass 段 + DOM needle 区的联合覆盖需求,`2026-08-23_scoring_function_study.zh.md` §6.10/§9 B3072)。
- **③(wn)是把两者接起来的桥**:没有它,Quest 上界在变长 chunk 上系统性偏向大块(同一 Quest 打分,fixed16 98 → tree 89,p=0.02;token 预算口径 needle 0.20 vs 0.80),tree 几何就不可用,上面的效率也就拿不到。α=0.25 是 §5.5 离线扫出的最优。
- 所以准确的表述是:**mixmax_wn 的贡献是"使能"而非"超越"**——它让 web-agent 的结构化几何达到 fixed-page 基线的准确率(从显著落后拉到统计不可区分),效率收益随几何而来;它从未在准确率上打败 Quest/Block(报告从头到尾没有这个 claim)。

## 5. 其他 post-Quest 方法在这两个轴上做了什么

两个正交的轴:**单元几何**(固定 / 可变)× **head 聚合**(逐 head / 池化)。文献定位(basis:`2026-08-23_scoring_function_study.zh.md` §8.5 的调研 + 公开论文;标 ◇ 的是文献判断、未在本工作实测):

| 方法 | 单元 | 摘要 | head 处理 | 长度归一 |
|---|---|---|---|---|
| Quest | 固定 16-tok page | min/max envelope | 逐 head,max | 不需要(定长)|
| Vortex BlockSparse | 固定 page | centroid+softmax | 逐 head,max | 不需要 |
| ArkVale ◇ | 固定 page | bounding-volume digest | 逐 head 类 | 不需要 |
| LServe ◇ | 固定分层 page | 分层摘要 + **时间复用**(我们 TSA_RESELECT_K 的出处)| — | 不需要 |
| ShadowKV ◇ | 固定 chunk | mean-pooled landmark(centroid 类)| 低秩/池化 | 不需要 |
| AdaKV / 各类 per-head budget ◇ | 固定 | — | **反方向**:更逐 head(每 head 独立预算)| 不需要 |
| SeerAttention ◇ | 固定 block | **学习的** pooled-Q 门控(池化 + 训练补偿)| 池化 | 训练内隐 |
| InfLLM / Landmark ◇ | 固定 block | 代表 token / 学习 landmark | 各异 | 不需要 |
| ClusterKV / PQCache ◇ | **可变**(key 空间聚类)| centroid | 各异 | 无(centroid 无膨胀偏置,但吃 dilution)|
| PBS-Attn / Louver | 置换 / range-search | — | — | 不适用 |
| DHSA (arXiv 2510.24606) ◇ | **可变**(自适应分段) | 聚合 embedding(centroid 族) | — | **有:√L 缩放**(直接先例) |
| AB-Sparse (arXiv 2605.12110) ◇ | 自适应 block size(**按 head 分配,head 内均匀**) | Quest 类 | 逐 head | 不需要(同一次选择内定长) |
| **本工作** | **可变(输入结构/DOM 切分)** | envelope | 组内 mean | **(16/L)^α,只作用于宽度项** |

规律修正(2026-08-28 检索后):固定单元仍是主流,但**"变长单元 + 长度归一化"并非无人考虑**——DHSA(2025-10)对自适应变长 chunk 明确做了长度归一(把聚合 embedding 按 √chunk_size 缩放消长度偏置);AB-Sparse 按 head 自适应选 block size(head 内仍均匀,故其打分不需跨长度归一),并复证了 Quest 在大 block 下掉精度。因此本工作在归一化上的**可辩护窄声明**只有:对 min/max envelope 上界做"中心 + 宽度"分解、**只归一随长度几何膨胀的宽度项**((16/L)^α,α 在真实 dump 上调优)——DHSA 归一的是 centroid 族的整体表示,envelope 上界的膨胀偏置及其针对性修正暂未查到先例(限于检索覆盖,保留被指正的可能)。head 聚合方面结论不变:两派都有人用,本工作的增量是控制变量 ablation + 分预算区间的失效机制。

## 5.5 两个必须直面的追问(2026-08-28 补)

**"head 平均是你们的贡献吗?"——不是。** GQA 组内 query 池化是部署系统的常见做法(摘要按 KV head 存,池化是自然选择;TSA 部署代码本来就这么做),也存在反方向流派(Quest 逐 head max、AdaKV per-head 预算)。本工作可主张的只有证据:把 head 聚合当独立变量的控制变量 A/B(≥3k 预算下 mean 与 max 端到端不可区分,n=190)+ 失效边界的机制刻画(<3k 时 retrieval-head 稀释;head-max 0.537 vs head-mean 0.479 于 failure-enriched dump)。

**"把 Quest 的 page 调大不是同样减少单元数吗?"——会,但要交不同的税,且该对照未在本 workload 实测。** ① 均匀大 page 在每一处付"粒度税":固定预算下选 64-token page 拿 5-token needle 浪费 59 token 的覆盖,B2048 的实测(细粒度赢)与 Quest 论文自己的 page-size ablation(16 最优)都指向这一点;变长 chunk 只在语义连贯处大、且边界与元素对齐(fixed 大 page 会把 [idx] 与元素文本切开,选一个元素花两页预算)。② 病理不同:均匀 page 无相对大小偏置(wn 在其上是恒等变换),病在上界整体变松 + 粒度税;变长 chunk 的病才是相对偏置,才需要 wn。③ 缺口:fixed-64/32 同预算对照没有本地实测,是应补的 ablation(离线 dump 版可先行)。

## 5.6 打分公式的部件溯源(2026-08-28,逐部件判定 novelty)

| 部件 | 内容 | 先例 | novelty 判定 |
|---|---|---|---|
| (a) 摘要统计量 | 每单元每 KV head 的逐维 min/max envelope | Quest(page metadata)、ArkVale(bounding-volume digest)、LServe(kmax/kmin 附在物理页尾) | 无 |
| (b) 上界打分 | Σ_d max(q_d·M_d, q_d·m_d);"中心+宽度"只是它的恒等改写 | Quest 原样 | 无 |
| (c) GQA head 池化 | 组内 mean → KV head 间 mean | TSA 既有;部署系统常见;AdaKV 反向 | 无(增量仅为 ablation 证据) |
| (d) 宽度项长度归一 | 宽度项 × (16/L)^α,α=0.25 实测调优;中心项不动 | **未见**(检索限度内) | **是,且是通用技术而非 web 特有** |

(d) 的技术内核:偏置来自**极值统计**——盒半宽是 L 个 key 的逐维 max/min 统计量,iid 近似下随 L 以 ~√(2 ln L)·σ 的**次线性(log 型)**速率增长;中心(midrange)无系统性增长。因此修正只该作用于宽度项、且应当平缓——幂律 (16/L)^0.25 在 L∈[16,256] 上近似 log 修正(α=0 大块霸榜、α=1 过惩罚,§5.5 扫描)。与 DHSA √L 归一的本质区别:DHSA 修的是 **mean/sum 聚合表示**(O(L) 统计量),对象、增长律、修正位置(整体 vs 仅宽度)全都不同。与 Tactic/ClusterKV 的对照:同为变长单元(K-means 聚类),但走 centroid(天然长度不变)+ **采样/分布拟合**补 centroid 误差——绕开了 bound,也就绕开了归一化问题,代价是 dilution(centroid 族在 token 预算下 needle 0.06–0.10 vs 带宽度项 0.80,§5.2)。

**"变长 × 摘要类型"的格子图**:fixed × bound = Quest/ArkVale/LServe;variable × centroid(+采样修正)= ClusterKV/Tactic/PQCache;variable × aggregate + √L 归一 = DHSA;**variable × bound + 宽度归一 = 本工作(空格)**。占这个格子的理由链:要结构对齐的变长单元(web 动机)∧ 要 needle 敏感(agent 要点中具体元素)⇒ 必须 bound(centroid 会 dilution)⇒ bound 在变长下有膨胀偏置 ⇒ 必须宽度归一。**(d) 与 web agent 的关系是动机与证据,不是技术边界**——代码 AST 块、markdown 章节、日志记录等任何结构变长切分都可直接用;若要把 (d) 升格为独立方法主张,需要 ≥1 个非 web 变长设定的验证(未做,见 §5.7 改进项)。

## 5.7 Research contribution 的定稿评估(2026-08-28,工程修复不计)

**逐条判定(检索后):**

| 候选 | 判定 | 依据 |
|---|---|---|
| (1) DOM 语义切割作为 KV 选择单元 | **可主张(项目级),本工作的贡献是"使能 + 定量刻画"** | 文本级 DOM 剪枝已是拥挤赛道(Prune4Web、FocusAgent、Observation-Reduction 综述);GUI 侧有 agent 专用 KV **压缩/驱逐**(GUI-KV 等,vision token 侧);但 **DOM 结构对齐的 query-aware 动态 KV 选择**未见先例。本工作补的是:让它达到 fixed-page 基线(打分使能)+ 3k 边界与双机制的定量刻画。注意 caveat:精度是追平不是超越,fixed-64 对照未测。 |
| (2) 打分公式改进 | **窄声明成立**:变长单元上界的宽度项长度归一化 | "中心+宽度"分解本身是初等恒等式(max(a,b)=(a+b)/2+\|a−b\|/2 逐维应用),Quest 的公式就是它的另一种写法、LServe 每页也存 kmax/kmin——**分解不是贡献,它只是指出偏置住在哪一项的透镜**。贡献 = 识别"bound 类打分在变长单元上的长度膨胀偏置"+ 只归一宽度项的修正 + 真实 workload 验证。DHSA 归一的是 centroid 族聚合(√L),bound 的宽度项归一未见先例(hedged)。head-mean 不是贡献(用户判断正确)。 |
| (3) 跨 token 选择复用 | **不是贡献** | LServe(2502.14866)的 "reusable page selection" 就是它:选择器只在 chunk 起点激活、后续 token 复用,选择开销 ÷4、无精度损失。我们的 rs8 只是在新 workload 上的复现(Jaccard 0.81/0.71 + 55 vs 58)。它也不是 web-agent 特有的。 |

**"每个点都必须是 web-agent 痛点"检验**:真正 web 特有的只有——DOM 结构几何(1)、prompt 结构决定的 3k 阈值与 sys/DOM 双区张力、元素级失效税收学(相邻 index、search-input 子树)、以及**跨轮 DOM 突变下的复用机会**(未做)。打分(2)是服务于(1)的通用组件;复用(3)是通用技术。

**如果不够,怎么改进(按杠杆排序):**

A. **跨轮(cross-turn)复用——最强方向,web 特有且无人做**:agent 每步重新 prefill 6–14k token(prefill 才是 web-agent 时延主项,decode 只 ~1k);逐字节前缀缓存只覆盖 ~59%,其余是"内容未变但位置平移"的 DOM 子树(RoPE 使 naive 复用失效)。语义 chunk 的**跨轮身份**(fixed page 在页面编辑后没有稳定身份,tree chunk 有)使两条路可走:轻量版=跨轮**选择先验**迁移(上一轮的 needle chunk 本轮先验加权,零位置问题);重量版=chunk 粒度的 CacheBlend 式位置修复复用 prefill KV。这是"语义切割才能解锁"的独有卖点。
B. **用结构赢下紧预算区**:sys/DOM 双区张力(§6.10)指向混合几何——DOM 区用 fixed16 细粒度(needle 精度)+ 指令/历史区用粗语义块(mass 覆盖),目标是在 ≤2k 同时超过纯 fixed16(55)与纯 tree(≤50),把"追平"变成一个区间的"超越"。
C. **欠账 ablation**:fixed-64/32 同预算对照;在线 task-success(teacher-forced agree 之外);head-max-on-tree 已测(≤.515,记录在案)。
D. **定位声明**:与 prompt 级 DOM 剪枝(Prune4Web/FocusAgent)正交可组合——KV 级选择逐层逐步、不删信息、可逆,且不需要额外 LLM 调用。

## 6. 结论(诚实版)

mixmax 很通用——**通用到它不该被叫作 web-agent 的打分创新**。正确的账目是:Quest 贡献了上界公式;TSA 的 kernel 结构决定了 head-mean(实证表明在 ≥3k 的工作区间免费,并省 8× 打分计算);web agent 贡献的是**几何**(DOM 变长 chunk,5–7× 单元数)和**工作点**(预算 ≥3k、prompt 结构决定的 ~3k 阈值);wn 是让前两者兼容的、文献里因"大家都用定长单元"而缺位的小修正。它的验证边界也要说清:一个模型(30B-A3B,GQA 8)、一个 workload(浏览器轨迹 teacher-forced replay)、n=132–190 的配对功效;head-mean 在其他 GQA 宽度/其他任务上的免费性,以及 <3k 区间的 head-max 优势是否值得做成混合聚合,都是没做的实验。

### 6.1 Contribution 清单(2026-08-28 定稿,按可辩护程度排序)

1. **诊断并修复两个使 TSA 全部历史结论失效的实现 bug**(flat-KV 布局错读、1024 页容量崩溃),均有最小 reproducer;前者促成 main 的分离 K/V pool 结构性修复,后者已合入(修掉 main 上标为 "root cause still open" 的崩溃)。
2. **证明"snapshot 语义变长几何"在打分层面可行**:刻画两类标准打分在变长单元上的对偶失效(envelope 的 width bias / centroid 的 dilution)+ TSA 原 2corner 的绝对值退化;给出组合 mixmax_wn,使 tree 几何首次与 fixed16 基线及 full attention 统计不可区分(KV ≥3k),同时打分单元 5–7× 少、select 快 1.3–1.55×。
3. **envelope 宽度项的长度归一化这一具体形式**(见 §5 规律修正的窄声明;"变长单元要归一"的一般想法有 DHSA 先例,BM25 式长度归一更是 IR 常识)。
4. **边界与方法论**:预算三区间(≥3k 透明 / ~3k 交叉 / ≤2k fixed16 胜,附双机制 case study)、head 聚合的控制变量 ablation、fidelity-to-dense / failure-enriched dump / 离线代理指标陷阱等评测方法学。
5. **系统件**:打分的 CUDA kernel 移植、时间复用(TSA_RESELECT_K)及其适用边界、微基准。

明确**不是**本工作的 contribution:head-mean 池化(TSA 既有 + 业界常见)、envelope 上界公式(Quest)、语义树切分本身(TSA 项目既有)、时间复用思想(LServe)。

**指针**:公式与 toy example → `2026-08-23_scoring_function_study.zh.md` §1.1;head-mean 失效 case → §6 案例 1、§6.10、§9 cand_sweep2b;wn 的来历与 α 扫描 → §5.3/§5.5;端到端矩阵 → §4.1、§9 Stage C/D;效率数字与运行方法 → `BrowserSparseAttention` README 的分支章节。

### 附注:quantization 的位置(避免与"定量刻画"混淆)

本方法唯一的数值量化是 selector 的 chunk 摘要(envelope M/m、centroid)按 TSA 既有工程存为 fp8(e4m3 + per-vector scale),动机是摘要在每 decode 步每层都被整读一遍、fp8 减半该流量;KV cache 本体始终 bf16,选中 page 的 attention 用全精度 K/V。它只影响排序、实测不引入可见病理(2026-08-23_scoring_function_study.zh.md §5),不构成任何 contribution 主张。对照:query-aware 选择类方法(Quest/LServe/ArkVale)摘要普遍为 fp16、无量化主张;KV cache 量化(KIVI/KVQuant/GEAR)是独立且可组合的另一族方法。

## 7. 混合切块方式:在线验证结果(2026-08-28/29)

**扩规模终值(half1 190 步,n=189 可用,2026-09-01)**:hyb_sf25 agree **94 vs fixed16-Quest 91(+3;配对 18W/15L,p=0.73,CI [−8,+14])**,旧 tree 82,dense 97;valid 156(最高)、none 32(最低,低于 dense 的 35)。两个规模的点估计(+2/+3)方向一致、均不显著;与 dense 差 −3 在噪声带内。**速度(CUDA mixmax kernel 实测)**:混合切块 select **51.7–65.0 µs/次(53/239 单元,T=6.2k/23k)vs fixed16 74.1–114.0 µs(424/1672 单元)= 1.4–1.75×**。至此 accuracy(两规模追平 dense、稳定小胜 fixed16 点估计、none/valid 全场最优)与 speed(select 1.4–1.75× + 单元 4.6× 少)两条证据线闭合;分支已推送(README 含完整 knobs/结果/bench,HEAD 3bbb01b,作者 shiqihe@umich.edu)。

**小规模终值(strat20 132 步,超时全部补跑,`analyze_stageD.py` 配对)**:hyb_sf25 agree **57 = dense(57)> fixed16-Quest/Block(55/55)**,配对 14W/12L(p=0.85,+2 为点估计、不宣称显著);**valid 112、none 19 为全场最优(含 dense)**;打分单元 159 vs fixed16 的 732。hyb_sf35 = 56。旧 tree 的 48 → 57:紧预算区完全收复。**half1 扩规模中期(101/190 可用步)**:hyb_sf25 51 > Quest 50 > 旧 tree 43(dense 56)——+1 点估计与小规模方向一致;剩余 89 步补跑中(第一次被自设的 abort 阈值误杀 + 补跑腿 OOM 竞态,阈值已改为 MAX_ERRS 可配、静置 300 s)。

**预算分配轴的定位(2026-08-29 检索)**:现有 budget-allocation 工作在 head 轴(Ada-KV)与层轴(PyramidKV);粒度×预算的耦合分析已出现在 head 轴(arXiv 2605.07719 的 streaming/retrieval head 区分)。**按 prompt 语义区域(指令 vs 观察)分配粒度与预算保底,未见先例**——这与 §5.6 的"variable × bound + 宽度归一"共同构成本方法的三个空格。

设计(§5.7 路线 B 的落地):**按 web-agent prompt 解剖分区的几何**——DOM observation 区按元素切(合并 ≥6、切分 ≤64 token),指令/历史区粗语义块(≥64/≤256),全体在同一 mixmax_wn 榜上按 token budget 竞争(跨尺寸可比性由宽度归一保证,wn 由此成为承重件),另有 sys-floor 保底指令区。离线 v3(23 个 failure-enriched dump,已建模 always-include、sysCov 扣 sink):B2048 needle **0.515 vs fixed16-Block 0.445**(+16%),打分单元 **159 vs 732**(4.6× 少),代价指令区覆盖 0.27 vs 0.50;B4096 needle 0.797 vs 0.642。离线无法仲裁"指令覆盖低是否触发 none/空转"——正在 online replay(strat20 132 步,sf∈{0.25,0.35},对标 fixed16 的 55)。若 >55:首个准确率优势,叙事升级为"prompt 解剖驱动的区域几何 + 元素级叶子 + 跨尺寸归一排序";若否:按 none 率迭代 floor/粒度。分层 B&B 下降(hier_bb)离线未优于平铺混合(evals 170 vs 112)暂缓。

