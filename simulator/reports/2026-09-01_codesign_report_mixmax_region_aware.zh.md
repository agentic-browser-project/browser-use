# Web-Agent 稀疏注意力报告:mixmax_wn → Region-aware Chunking → Co-design

**日期**:2026-09-01(实验窗口 08-14 → 09-01;仍在运行的实验标注 *in progress*)
**模型**:Qwen3-VL-30B-A3B-Instruct(48 层,32 query heads / 4 KV heads,GQA group size = 8,head_dim = 128),纯文本
**硬件**:NVIDIA GB10(sm_121,119 GB unified memory),spark00
**代码**:`BrowserSparseAttention` 分支 `shiqihe/region-aware` @ f2b7f09(基于 main;mixmax 部分已由 PR#5 合入 main,旧分支 `shiqihe/mixmax-main` 已删除)
**配套文档**:`2026-08-23_scoring_function_study.zh.md`(scoring 与 bug 的完整细节)、`2026-08-28_mixmax_head_aggregation_design.zh.md`(contribution 论证与 related work 对照)

---

## 术语表(阅读本文需要的全部概念)

| 术语 | 定义 |
|---|---|
| **KV cache** | 生成时,prompt 中每个 token 在每层都存一对 key/value 向量;full attention 下每生成一个 token、每层都要读取全部 KV。 |
| **sparse attention** | 每个 decode step、每层只读取 KV cache 的一个子集,以降低延迟。 |
| **chunk** | 把 prompt 的 token 序列划分成的连续区段,是"选读哪些 KV"的基本单位。Quest 等方法用固定 16-token chunk(称 page);本工作用变长语义 chunk。 |
| **token budget(B)** | 每个 decode step、每层允许读取的 KV token 总数上限(如 B=2048)。越小越稀疏、越省时。 |
| **min/max envelope** | chunk 的摘要:对 chunk 内所有 key 向量逐维取最大值得向量 M、最小值得向量 m。 |
| **upper bound scoring** | 用 envelope 计算 Σ_d max(q_d·M_d, q_d·m_d):可证明 chunk 内**任何** key 与 query q 的点积都不超过该值(故称 upper bound),回答"这个 chunk 最好情况下能多相关"。 |
| **width term** | 上式等价改写 q·(M+m)/2 + Σ_d\|q_d\|·(M−m)_d/2 的第二项;(M−m) 反映 chunk 内 key 的分散程度,chunk 越长该项天然越大。 |
| **centroid scoring** | 另一流派:用 chunk 内 key 均值打分(q·mean),回答"平均多相关"。 |
| **reference element / needle** | offline 评测中参考轨迹该步实际点击的页面元素(如 `[54]<input …>`)。**needle hit rate** = 该元素所在 chunk 被选进 budget 的比例;未选中则模型不可能输出正确编号,是行为正确的必要条件。 |
| **attention mass** | full attention 的 softmax 权重;一个区域的 mass = 区内 token 权重之和。**mass coverage** 衡量被选 chunk 对 full attention 的逼近。mass 多集中在序列开头与指令上,needle 往往只是 DOM 里权重很小的一行——两个目标经常冲突。 |
| **attention sink** | 序列开头几个 token 吸收大量权重的现象;所有方法都永远保留它们(always-include)。 |
| **agree** | teacher-forced offline replay 指标:把参考轨迹某步的原始 context 喂给被测配置,输出元素编号与参考一致记 1;分母只含 index-action 步。不等于在线任务成功率,但同批步骤可严格配对。 |
| **run-to-run variance** | full attention 自身重跑也会改变约 20% 输出(decode 随机性);n≈130–190 时小于 5 步的差异不可解读,所有比较用配对 sign test + bootstrap CI。 |
| **GQA / head pooling** | 32 个 query head 每 8 个共享 1 个 KV head(4 组)。打分时每组的 8 个 query head 可平均为一个向量(head-mean,便宜)或逐 head 打分取最大(head-max,保留只在少数 head 出现的信号,贵)。 |
| **budget floor** | 本工作引入的超参:把 budget 的固定比例预留给某个 prompt 区域,区域内部再按分数选。记号 sf25 = system-prompt floor 0.25。 |
| **准入(admission)** | 打分之后的装填步骤:候选 chunk 按分数从高到低,依次装进 budget(装得下就装,装不下跳过下一个),直到预算用完——即贪心背包。floor/分区只是把这个过程限制在某区域的专属预算内先做一遍。 |

---

## 0. 主要结论(方法演进总表)

评测集:**strat20**(20 任务 / 132 index step)与 **half1**(50 任务 / 190 index step,strat20 是其子集);同批步骤配对比较。

| 方法版本 | agree @B2048(132 步) | agree @B2048(190 步) | agree @B1024(132 步) | chunk 数 |
|---|---|---|---|---|
| full attention(参照) | 57 | 97 | — | — |
| fixed-16 baseline(Quest / Block) | 55 / 55 | 91(Quest) | **55**(修正后,n=132,none 31,0 超时) | 732 |
| ① mixmax_wn + **单一参数**变长 chunk(全 prompt 一套合并/切分参数 16–256) | 48 | 82 | — | ~120–250 |
| ② mixmax_wn + **分区域参数**变长 chunk(region-aware)+ floor | **57** | **94** | **53**(n=132 全清,valid 123 / none 8;与 baseline 配对 = **平**) | ~159 |
| ③ hierarchical selection(树上下降) | offline 即劣于②,未上线 | — | — | — |
| ④ co-design(③ 的替代:预算三区划分 + 交互元素优先 + DOM head-max) | **52**(干净步 115;配对 vs ② 13W/12L = 平;**17 步超时跑不完**) | — | **46**(n=128)——配对 vs ② 14W/20L,**真准确率赤字** | ~159 |

三条结论:**(a)** 只修 scoring 在 B≥4096 追平全部 baseline,但 B2048 仍落后(48 vs 55)——瓶颈在 chunking 与 budget 分配;**(b)** ② 在同 setting 下比 ① 提升 +9(132 步)/+12(190 步),B2048 达 full attention 水平、B1024 与 baseline 配对打平(13W/14L)且格式遵从显著更好(none 7 vs 31),chunk 数为 fixed-16 的 1/4.6、selection latency 低 1.4–1.75×——**② 是本工作的最终可交付配置**;**(c)** ④ online 否证定稿:B1024 真准确率赤字(46,vs ② 配对净 −6),B2048 干净步与 ②/baseline 配对全平但 17/132 步跑不完(吞吐失败),两个 budget 均无胜场;offline needle 代理两次方向预测反(§4.5/§6.2)。

---

## 1. 背景

### 1.1 web agent 的一步与 prompt 结构

browser agent 每步把如下 prompt 交给 LLM,要求输出动作(JSON,含目标元素编号):

```
[system prompt]   ~5.3k token:操作规则、输出格式、示例      —— 不可能成为动作目标
[任务 + 历史]      数百至 1.3 万 token:目标与此前步骤        —— 同样不可能成为动作目标
[DOM observation] 0.7k–6k token:页面元素列表,一行一个元素
                   例:[54]<input Search …>                  —— 动作目标(needle)在这里
[URL 等收尾]       几十 token
```

context 总长 6k–23k token;decode 阶段 full attention 每 token 每层读全部 KV,是延迟主项之一。

### 1.2 sparse attention 的两个设计轴

把 KV cache 切成 chunk,每步每层只读 top 的一部分。两个决定性选择:**chunking** 与 **scoring**(upper bound 还是 centroid,见术语表)。centroid 的弱点是 **dilution**:138-token chunk 里只有一行是目标时,均值把它稀释到 1/138(实测此类 chunk 被选中率仅 1%)。upper bound 的强项是单个强匹配 key 就能抬高整个 chunk;弱点见 §2.1。

本文出现的 chunking 只有三种,先钉死命名(后文 ①②④ 一律指下表):

| 名称 | chunk 是否变长 | 规则 |
|---|---|---|
| **fixed-16**(Quest/Block baseline) | 否,一律 16 token | 每 16 个 token 切一刀,与语义边界无关 |
| **①:单一参数变长 chunk** | **是** | 按语义树切,**全 prompt 一套参数**:相邻小节点合并到 ≥16 token,大节点切到 ≤256 |
| **②/④:分区域参数变长 chunk(region-aware)** | **是** | 同样按语义树切,但**参数随区域不同**:指令/历史区合并 ≥64、切 ≤256;DOM 区合并 ≥6、切 ≤64(≈一行元素一块) |

也就是说 ① 和 ②④ **都是变长 chunk**;区别不是"变长 vs 均匀",而是切分参数是否随 prompt 区域变化。

### 1.3 方法学

所有 agree 在同批步骤配对,报告 sign test 与 bootstrap;低于 run-to-run variance 的差异不解读。offline 代理指标(needle/mass)只在**失败案例富集的真实 query 流** dump 上使用——普通样本上它曾把 B2048 的方向预测反(`2026-08-23_scoring_function_study.zh.md` §9)。

---

## 2. 第一部分:scoring(mixmax_wn)

### 2.1 问题:TSA 原 scoring 的退化

TSA 的出发点是按 DOM/ChatML 树切**变长语义 chunk**(16–256 token)。但其部署的 "2-corner" scoring 是 max(q̄·M, q̄·m)——先逐维求和再取 max。当 q̄ 各维正负混合时,与恒正的 (M−m) 相乘的各维贡献互相抵消,width term 实际消失,**退化为只比较 chunk 中点**,丢掉对单个强匹配 key 的敏感性。(历史上另有两个实现 bug——KV 布局错读、1024-page 容量越界——均已修复,见 `2026-08-23_scoring_function_study.zh.md` §3。)

### 2.2 mixmax_wn 的构成与归属

```
2-corner(TSA 原版): s = max( q̄·M , q̄·m )                          ← width term 相消
mixmax:              s = q̄·(M+m)/2 + Σ_d |q̄_d|·(M−m)_d/2           ← 即 Quest 的 upper bound
mixmax_wn:           s = q̄·(M+m)/2 + (16/L)^0.25 · Σ_d |q̄_d|·(M−m)_d/2
```

| 组成 | 来源 | 本工作的贡献? |
|---|---|---|
| upper bound 公式(逐维) | Quest 原样 | 否 |
| GQA head-mean(组内平均 + KV head 间平均) | TSA 既有 / 业界常见 | 否(仅提供消融证据:B≥3k 时与 head-max 无端到端差异) |
| **width term 长度归一化 (16/L)^α** | 本工作 | **是** |

**从零开始:一个 chunk 的分数是怎么算出来的**(最小例子,2 维、2 token、1 个 KV head 组)。prompt 里每个 token 存的是 key(每层每 KV head 一份);query 是正在生成的新 token 产生的。chunk 打分没有任何"逐 token 打分再平均"——那是 centroid 派的思路——而是:(1) 预处理把 chunk 压成两个向量:k₁=(2,3)、k₂=(0,−1) 逐维取最大/最小得 M=(2,3)、m=(0,−1),原始 key 从此退场;(2) 组内 query head 平均:q_a=(1,1)、q_b=(1,−3) → q̄=(1,−1);(3) 逐维挑对 q̄ 有利的端点:第 1 维 q̄₁>0 取 M₁ 得 1×2=2,第 2 维 q̄₂<0 取 m₂ 得 (−1)×(−1)=1,分数=3——框内任何 key 与 q̄ 的点积不超过 3(实际 q̄·k₁=−1、q̄·k₂=1 ✓);注意最优角落 (2,−1) 混合了 k₁ 的第 1 维与 k₂ 的第 2 维,框里并无此 token,所以是乐观上界,chunk 越长越乐观(→ 宽度归一化);(4) 4 个 KV head 各得一个分数后取平均;(5) 每层每步对全部 chunk 算出一个数,排序、按预算准入。

**为什么公式里"没有 min"(与 Quest 逐维公式的等价性)**:Quest 写作 s = Σ_d max(q_d·M_d, q_d·m_d)——min envelope m 明确在场,每一维在两个端点里挑大的(线性函数在区间上的最大值必在端点:q_d≥0 选 M_d,q_d<0 选 m_d)。用标量恒等式 max(a,b) = (a+b)/2 + |a−b|/2,代入 a=q_dM_d、b=q_dm_d 并利用 M_d−m_d≥0,得逐维等价形式 q_d·(M_d+m_d)/2 + |q_d|·(M_d−m_d)/2;对 d 求和即 mixmax 的"中心 + 宽度"式。**m 没有消失**:它在中心 c=(M+m)/2 与半宽 w=(M−m)/2 里;端点选择被绝对值自动完成。数值验证(d=2,q=(2,−1),M=(3,5),m=(1,−4)):Quest 逐维 = max(6,2)+max(−5,4) = 10;mixmax = q·c + |q|·w = 3.5+6.5 = 10,逐 bit 相等;而 TSA 原版整向量 2-corner = max(q·M, q·m) = max(1,6) = 6 < 10——各维角选择被迫一致,bound 变弱,即 §2.1 的退化。采用改写形式的两个工程理由:(a) width 项被单独暴露,(16/L)^α 归一化才有落点;(b) c/w 每 chunk 预计算一次,打分只需两个 einsum(`_score_mixmax` 的 ctr/wid 两行)。

**head 维度的完整数据流**(公式里只写了一个 q̄,容易漏看第二次平均,这里写全):

```
32 个 query head
  →(第一次平均)GQA 组内 8 个平均成 1 → 4 个 q̄_h(tree_sparse_selector.py:1059)
  → 每个 q̄_h 与自己 KV head 的 envelope 算 upper bound → 每 chunk 4 个分数
  →(第二次平均)对 4 个 KV head 的分数取 mean → 每 chunk 1 个分数(同文件 :1288,
     `(ctr + wid).mean(dim=1)`;centroid 路径 :1105、2-corner 参考路径 :1141 同样收尾)
  → top-k / budget 准入 → 该层全部 32 个 head 共用同一份选中集
```

两次平均都是**有意设计,不是 bug**:函数 docstring 明写 "Mean over KV heads";代码还留有替代聚合方式的开关(`BSA_HEAD_COMBINE=adakv`,per-head softmax 后求和);CUDA kernel 复现同样的数学(与 torch 路径选择结果 Jaccard 0.99–1.00)。设计原因:serving kernel 每层只 gather **一份**共享的 page 集合,所以 4 个 per-head 分数必须先归并成一个标量才能排序。**与原版 Quest 的真实差异在此**:Quest 论文是**每个 head 各选各的 top-k page**(不做跨 head 归并,代价是选择状态与 gather 都乘以 head 数);我们 harness 里的 fixed-16 "Quest"/Block baseline 用的是 Quest 的打分公式 + 与我们相同的"head-mean + 共享选中集"约定——所以内部对比在这条轴上是公平的,但对外表述应写成 "Quest-style scoring",不是逐字复现 Quest(§6.5)。tight budget 下这个 mean 确实会稀释只在少数 head 上的检索信号——这正是 ④ 给 DOM 块换 head-max 的动机(§4.1,实现在同文件 :819–:829,`torch.where(self._dom_mask, hm, scores)`)。

**共享选中集是权衡,不是绕不开的必然**(Quest 证明绕得开,代价在 kernel 侧)。以 16k context、page 16、B=2048 为例:
- **共享集(我们)**:打 1024 个 page 的分 → 1 次跨 head 归并 → 1 次 top-128 → 1 张 128 项页表 → 1 次 32-head paged-attention 调用。收益:(i) 兼容 sglang/vLLM 的页表契约——主流 serving 引擎的 block table 是 **per-sequence** 的,没有 per-head 页表的 decode kernel;(ii) 选择路径的 top-k/索引构建/reselect 缓存都是 1 份而非 4 份——我们的 selection 本来就是 launch-bound(51–114 µs 量级),乘 4 会吃掉 1.4–1.75× 的优势;(iii) GQA 8:1 下若拆成 4 次 "8 个 query head" 的 attention 调用,单次调用 head 数太少,GPU 占用率变差。
- **per-head 集(原版 Quest)**:每个 head 自己的 top-128、自己的页表。真实收益:4 个 head 的选中集**并集**最多可覆盖 4×2048 个不同 token(head 各自特化,recall 上界更高),而每个 head 读的字节数与共享集完全相同——即精度上界换元数据/kernel 复杂度。Quest 能这么做是因为它自带研究原型 kernel(FlashInfer 扩展,原生 per-head 页表),不受 serving 引擎接口约束;per-head 驱逐类方法(H2O/SnapKV)同理,直接改 attention 实现。
- 另注:即便要共享集,mean 也不是唯一归并方式——代码里有 `adakv` 开关(per-head softmax 后求和),④ 对 DOM 块用峰值型聚合。"必须做某种跨 head 归并"只在"要共享选中集"这个前提下成立;完全 per-head 化的下一步工程量在 decode kernel。
- **发表侧的独立佐证(LessIsMore,arXiv 2508.07101,ICML26 投稿)**:该工作在 reasoning 模型上实测 ground-truth top-4K token 在各 head 间高度重叠(含 KV 组内重叠——这正是我们 query 组内平均损失小的经验依据),并据此把 per-head 独立选择替换为**跨 head 统一选中集**(各 head 用精确分数提 top-k 提名 → 按名次投票合并成一个共享集),GQA 消融(其附录 A.1.1)显示统一集 > per-head 独立(Head-to-Head)> 单 head 代表(Randomized),且选择频率越低差距越大——理由与我们相同:降低选择方差、跨层/跨步复用更稳、KV 访问简化。注意两点差异:(a) 它的归并是**名次联合投票**而非分数 mean,且分数来自在"选择层"顺带计算的**精确 full attention**(O(T),无摘要)——与我们"envelope 摘要 + 每层打分"的成本模型完全不同;(b) 它保底的是**时间近因窗**(budget 的 r=25% 给最近生成 token,基于"近因 token 占比恒定"的观测),与我们的 **prompt 区域保底**(sys/DOM floor)是同一设计模式在不同轴上的实例——web agent 场景生成极短、近因轴几乎空置,而 reasoning 场景没有 prompt 解剖结构,两者互补,共同支持"按结构先验保底预算"作为一般原则。

**归一化动机与 (16/L)^0.25 的由来**——分三步:

1. **理论定方向**:width term 由 chunk 内 key 的逐维极值决定,L 个样本的极值随 L 增长(iid 高斯近似 ~√(2 ln L)),因此**长 chunk 的 upper bound 系统性偏高**,与是否含目标无关;而中心项 q·(M+m)/2 不随 L 系统性膨胀。结论:需要一个只作用于 width、随 L 单调递减、增长缓慢的折扣。
2. **函数族与锚点**:选单参数幂律 (16/L)^α,锚定在 L=16(与 baseline page 同尺度的 chunk 折扣=1,分数尺度不动;L=64 → 0.707,L=256 → 0.5)。**为什么是 16/L 不是 1/L**:两者对 L 的依赖完全相同((16/L)^α = 16^α·L^{−α}),差的是一个全局常数,而总分是 ctr + f·wid 两项之和,这个常数实质上是**宽度项相对中心项的权重**(换 1/L 等于把宽度权重砍半,会改变排序)。一般形式是 s = ctr + λ·L^{−α}·wid,锚点选择 ≡ 选 λ:锚在 16 即 λ=16^α,含义是"参考粒度(baseline page size)上的 chunk 保持逐 bit 精确的 Quest 上界,长于它折价、短于它(6-token 元素块,(16/6)^0.25≈1.28)轻微升权",且不引入第二个自由超参;锚在 1 没有对应物(数据里不存在 1-token chunk),还得再补一个 λ 调回权重。**能否用动态锚(如 mean chunk size)**:固定锚之间只是 (λ,α) 的坐标变换(换锚 + 重扫 α 落回同一族);但 L̄ 依赖 prompt 的切块分布,会把打分从 (chunk, query) 的局部函数变成全局函数——跨步分数不可比(DOM 每步变异 → L̄ 抖动)、wn 因子不再是注册时可预计算的静态属性、消融归因混入隐式重标定;且方向相反:块普遍大 → λ=L̄^α 变大 → 乐观项权重升高,而块大恰是 bound 最松之时。锚 16 的操作含义:kernel page size = baseline 粒度 = bound 可信度被验证过的尺度,且保证"退化到 fixed-16 切块时打分 ≡ Quest 逐 bit"(消融卫生)。锚点未被单独扫描(与 α 共线);per-region 锚点是合法但未探索的低优先级旋钮。

**锚点消融(2026-09-04 实测,23 个失败富集 dump,同几何/同准入,仅换锚点)**:实测平均块长 79.3(范围 38–122)、中位 47.6,换成 L̄ 等于把宽度项整体乘 1.483。

| 锚点 | B1024 needle / sysCov | B2048 needle / sysCov |
|---|---|---|
| **A=16(部署值)** | **0.280 / 0.217** | 0.428 / **0.353** |
| A=32 | 0.270 / 0.208 | 0.424 / 0.343 |
| A=64 | 0.265 / 0.200 | 0.432 / 0.329 |
| A=median(L) | 0.271 / 0.204 | 0.434 / 0.336 |
| A=mean(L) | 0.264 / 0.198 | **0.437** / 0.326 |

B1024 上锚点 16 三项指标全胜(动态锚 needle −5.7%、sysCov −8.8%);B2048 上动态锚 needle 微涨 2% 但 sysCov 掉 7.6%(而 sysCov 正是 online none 率的主因)。固定锚 32/64 同样不优于 16,说明问题不在"动态 vs 固定"而在方向:块越长 bound 越松,此时应更不信任宽度项。结论:维持 A=16;要调宽度权重应改 α(有 β=0.245 的校准依据),而非换锚点。per-region 锚点仍是未探索的低优先级旋钮。α 当时在失败富集 dump 上离线扫描 {0, 0.25, 0.5} 确定:0 在线显著落后(89 vs 96,p=0.02),0.5 过度惩罚长 chunk,0.25 最优——**即 α 是拟合出来的超参,理论只给了方向与形状**。
3. **事后直接验证(2026-09-01 补测,`scoring_case_study/measure_width.py`)**:在真实 web-agent KV dump 上(2 组 dump × 3 请求 × 4 层,每档 150 个随机连续 span)直接测量 width term 随 L 的增长:

| L | 实测宽度(相对 L=16) | (16/L)^0.25 修正 | 修正后 |
|---|---|---|---|
| 8 | 0.767 | 1.189 | 0.912 |
| 16 | 1.000 | 1.000 | 1.000 |
| 32 | 1.227 | 0.841 | 1.031 |
| 64 | 1.449 | 0.707 | 1.024 |
| 128 | 1.656 | 0.595 | 0.985 |
| 256 | 1.858 | 0.500 | 0.929 |

拟合斜率 **β = 0.245**(逐 dump×层范围 0.192–0.279)——与部署的 0.25 几乎重合;修正后宽度在 L=8–256 全程平坦(±7%)。两点解读:(a) 当年用任务指标扫出的 α 与今天用几何量直接测出的增长指数一致,**超参不是碰巧,它就是数据里 width 的真实增长率**;(b) 实测 256-token 膨胀 1.86×,比 iid 理论的 √(ln256/ln16)=1.41× 更快——超出部分来自语义块的内容异质性(长 chunk 跨多个主题,envelope 被语义分散撑大,不只是抽样极值),这解释了为什么纯理论折扣(0.707)不够、扫描选中了更强的 0.5。文献缺位的原因:用 upper bound 的方法全部定长(Quest/ArkVale/LServe),用变长单元的全部 centroid(ClusterKV/Tactic;DHSA 对 mean 聚合做 √L 缩放,对象与增长规律都不同)。

### 2.3 实验结果

**B4096(half1,190 step,配对)**:fixed-16 Block/Quest 99/98,full attention 97,**①(单一参数变长 chunk + mixmax_wn)96**(与全部 baseline 差异不显著,p=0.65),同 chunking 无归一化 **89**(显著落后,p=0.02)。即归一化把变长 chunk 从显著落后拉进正常范围;B≥4096 是 sparse≈dense 的饱和区。

**B2048(strat20,132 step)**:① 仅 **48**,落后 fixed-16(55)。失败 step 的逐层选择记录复盘发现**两个互相冲突的失效机制**:
- **机制 A(粒度不足)**:目标元素位于 109–219 token 的大 chunk 中,被选中率仅 0.01–0.20(fixed-16 对应 page:0.15–0.66)——需要更细粒度;
- **机制 B(指令覆盖塌缩)**:system prompt 区 token 覆盖率跌到 7–14%,模型丢失输出格式约束,无动作输出暴涨(none 40 vs baseline 25–33)——需要保障指令覆盖。
固定 budget 下单一 chunking 无法兼顾两者 → 第二部分。

---

## 3. 第二部分:region-aware chunking

### 3.1 设计

按 §1.1 的 prompt 结构分区设定 chunk 粒度(解析树不变,只改叶子合并/切分规则):

| 区域 | 需求 | chunking | 理由 |
|---|---|---|---|
| system prompt + 任务 + 历史 | 只需 mass coverage(不含动作目标) | 粗粒度:合并至 ≥64、切分至 ≤256 token | 规则段整体相关或整体无关,粗 chunk 足够且打分便宜 |
| DOM observation | 动作目标所在 | **一行元素一个 chunk**:合并至 ≥6、切分至 ≤64 | 元素是最小语义单元;编号与其文本不分离 |

强调:**② 仍然是变长语义 chunk**,解析树、打分公式与 ① 完全相同;唯一区别是合并/切分参数从"全 prompt 一套(16–256)"变成"按区域两套(指令区 64–256,DOM 区 6–64)"。用 §4.2 的 toy DOM 四行(横幅 12 token、首页 6、搜索框 6、按钮 6)看三种 chunking 的切法差异:

```
fixed-16 :|[12]横幅12tok+[13]前4tok|[13]后2tok+[54]整行+[55]…|   ← 每16个token切一刀,元素行被腰斩,
                                                                  编号与其文本可能分属两页
①(≥16) :|[12]横幅+[13]首页 =18tok|[54]搜索框+[55]按钮 =12tok|…  ← 相邻行合并到≥16;真实网页中常是
                                                                  整个列表子树并成 100+ token 的大块
                                                                  (§2.3 机制 A 的 109–219 token 即此)
②(DOM≥6):|[12]横幅|[13]首页|[54]搜索框|[55]按钮|                 ← 一行一块;指令区仍用 64–256 粗块
```

所有 chunk 进入**同一个 mixmax_wn 排序**按 budget 贪心准入(见术语表"准入";6-token 元素与 256-token 规则段的可比性完全依赖 §2 的长度归一化);再加 **system-prompt budget floor**(见 §3.3)防机制 B;常规 always-include(sink + 最近 128 token + 已生成)照旧。典型 prompt 得 **~159 chunk**(fixed-16:732)。

**实现说明(与旧语义切分代码的关系)**——② 与 ① 共享 tree parser 与叶子收集(`_fix_parent_ranges` + `_collect_leaves`):"一行元素一块"不是新的切割机制,webarena parser 本来就把每个 `[idx]<tag>` 行解析成叶子,旧路径同样从这些叶子出发;改变的是叶子之后的合并/切分。但那段代码是重写(`_hybrid_chunks`,selector:1207)而非复用旧 `extract_leaf_chunks`(tree_parser.py:635),有四处实现差异需声明:
1. **区域边界用正则不用树标签**:`<|im_start|>user` 之后首/末个 `\[\d+\]<` 匹配定出 [dom_lo, dom_hi]。原因:树的 "observation" 标签在部分 dump 上不可靠(曾导致离线 sweep 的 DOM 区静默为空);代价:DOM 区是连续 token 区间,首尾元素之间的非元素行也按 DOM 参数切。
2. **gap 覆盖**:旧路径给叶子间隙(父节点属性 token)显式生成 gap chunk,保证 100% token 覆盖(实测 0 未覆盖);新路径的间隙只在被合并吞并时覆盖,实测 **~0.1%**(9/8482、9/7550 token)不属于任何 chunk、永不可选(部分落在 always-include 内)。判定:量级可忽略,列为 campaign 结束后的清理项(合并时顺延到下一叶子起点即可修)。
3. **合并溢出行为**:旧路径扩 buffer 前检查 ≤max 并提前 flush;新路径先合并(到 ≥min 即停)后统一按 max 切,罕见情形下"小前缀+大叶子"被焊接后按 256 任意偏移切开,边界对齐略差。
4. chunk label 不再保留(仅影响调试输出)。
同 prompt 实测 chunk 数:旧 (16/256) 104/90 个,② 92/78 个——**总打分单元数与 ① 相近**,差别是分布(指令区更粗、DOM 区更细);相对 fixed-16(数百上千)的单元数优势两者共享。

### 3.2 同 setting 对照:② vs ①(本节核心证据)

完全相同的评测集、budget、serving 配置,唯一变量是 chunking(+floor):

| setting | ①(单一参数变长 chunk) | ②(分区域参数变长 chunk) | Δ | 配对 | fixed-16 Quest/Block | full attention |
|---|---|---|---|---|---|---|
| strat20,B2048(n=132) | 48(valid 92 / none 40) | **57**(valid 112 / none 19) | **+9** | 17W/8L,p=0.108 | 55 / 55 | 57 |
| half1,B2048(n=190) | 82(valid 148 / none 35) | **94**(valid 156 / none 32) | **+12** | 29W/17L,p=0.104 | 91(Quest) | 97 |

说明:(a) 两行的评测集有包含关系(strat20 ⊂ half1),不能合并检验,但方向一致、量级一致;(b) ② 对 ① 的提升(+9/+12)远大于 ② 对 fixed-16 的差距(+2/+3,后者在 run-to-run variance 内),即把切分参数分区域后,变长 chunk 相对定长 baseline 的劣势被全部消除;(c) none 从 40 → 19(低于 full attention 的 33),机制 B 被 floor 直接治愈。

**中间消融(floor 单独的作用)**:在 ① 的单一参数 chunking 上只加 floor=0.05(不改 chunking):agree 48→**50**,none 40→**26**。即 floor 单独修复指令塌缩(机制 B),但 agree 仍落后 fixed-16 五步——大 chunk 里的目标元素仍选不中(机制 A),必须配合元素粒度 chunking 才到 57。两个组件各治一个机制,证据闭合。

**latency(GB10 实测)**:CUDA kernel 上 ② 的每次 selection **51.7–65.0 µs**(53/239 chunk,context 6.2k/23k)vs fixed-16 **74.1–114.0 µs**(424/1672 page),**1.4–1.75×**;kernel 相对 PyTorch 参考实现 24–54×;selection 结果每 8 步复用(思想来自 LServe)在 PyTorch 路径另省 2–4.8×,端到端 agree 无显著变化。

### 3.3 超参数一览(此前版本缺失,补齐)

| 超参 | 含义 | 尝试值 | 选定 | 依据 |
|---|---|---|---|---|
| α(wn_alpha) | width term 归一化指数 (16/L)^α | 0(不归一)/ 0.25 / 0.5 | **0.25** | 0 显著劣化(89 vs 96,§2.3);0.5 过度惩罚长 chunk,0.25 最优;事后在真实 KV 上直接实测宽度增长指数 β=0.245(0.192–0.279),与 0.25 重合(§2.2 第 3 步) |
| f(sys-floor) | budget 中预留给指令区(user turn 之前全部 token)的比例;区内按分数选,不足额时余量退回全局池 | 0.05 / 0.25 / 0.35 | **0.25** | online:0.05(50,配 ① chunking)/ 0.25(**57**)/ 0.35(56);0.25 与 0.35 差 1 步(variance 内),取更省的 |
| DOM 区 min/max | 元素 chunk 合并下限 / 切分上限(token) | — | **6 / 64** | offline needle 扫描:min 太小则 chunk 数爆炸,max 保证长元素行也不并入邻行 |
| 指令区 min/max | 粗 chunk 合并下限 / 切分上限 | — | **64 / 256** | offline mass-coverage 扫描;上限 256 与 kernel page 对齐 |

---

## 4. 第三部分:co-design

### 4.1 相对 ② 到底改了什么(逐轴对照)

用户问的四个问题直接回答:

| 设计轴 | ②(region-aware) | ④(co-design) | 变了吗 |
|---|---|---|---|
| scoring 公式 | mixmax_wn | mixmax_wn,完全相同 | **没变** |
| head 聚合 | 全部 chunk head-mean(4 组分数取平均) | **仅 DOM 元素 chunk 换峰值型聚合**(下称 DOM head-max;精确数学:32 个 query head 各自打分 → 同一组内位置的 4 个 KV head 分数**求和** → 8 个组内位置**取 max**,`tree_sparse_selector.py:826–:828`;严格的"32 头取 max"是另一个变体,离线消融排队中,见 §4.1.1);指令/历史区仍 head-mean | **变了(只在 DOM 区)** |
| chunking | 区域粒度(粗块 64–256 / 元素块 6–64) | 完全相同 | **没变** |
| hierarchical selection | 无 | **无**(树上 branch-and-bound 下降被 offline 否证后弃用,见 §4.4;④ 是 ③ 的替代而非叠加) | 均无 |
| budget 分配 | 两区:sys-floor 0.25 + 其余全局 | **三区:sys 0.25 / DOM 0.55 / 全局 0.20**,且 DOM 区内**可交互元素优先准入** | **变了** |

三个新组件的动机:
1. **三区划分**:head-max 只用于 DOM 后,DOM 块与指令块的分数尺度不同,不能同榜排序;划分后各区内部排序,尺度问题消失,同时 DOM 覆盖有了下限;
2. **交互元素优先**(action-space prior):先回忆"准入"= 按分数从高到低把 chunk 装进预算直到装满(术语表)。普通做法只看分数;④ 在 DOM 区的专属预算内把元素 chunk 分成两批——**第一批**:HTML 标签属于可交互类型的元素(`<a>/<input>/<button>/<select>/<textarea>` 等,即能被点击或输入的);**第二批**:纯展示元素(`<div>/<span>` 文本、横幅、图片说明等)。先把第一批按分数装完,预算有剩余才轮到第二批。依据是任务结构的一个事实:agent 输出的动作(click/type)只能落在可交互元素上,展示类元素永远不会是动作目标——预算紧张时,保住"全部可能的动作目标"比保住"分数更高的展示文本"更重要。判断只用元素的标签类型,不用任何参考答案信息;
3. **DOM head-max**:检索型信号常只出现在少数 query head("retrieval heads"现象),默认路径(query 分组平均 + KV head 间 mean)会把这种单头尖峰稀释;换成峰值型聚合(精确数学见 §4.1 表)后尖峰得以保留。元素 chunk 小而少,逐 head 打分开销可忽略。为什么不给指令/历史区也用——见 §4.1.1。

### 4.1.1 为什么指令/历史区**不**用 head-max

四个理由,前三个是机理论证,第四个是诚实声明:

1. **两个区域的优化目标不同,对应不同的正确聚合函数**。DOM 区要"找到那一根针":目标元素的信号可能只在一两个 head 上,聚合必须保峰,否则 needle 输给背景。指令/历史区要的是 **mass coverage**:一个 chunk 该不该进,取决于全部 head 加起来会往它投多少 attention 总量。总量是**对 head 求和**,而 mean 与求和同序(只差常数因子),是总量的正确代理;max 只反映最强的一个 head——某 chunk 在一个 head 上有尖峰、其余 head 全不理它,它对指令覆盖的贡献本来就小,max 却会把它排到前面。
2. **长 chunk 的 bound 松,max 是噪声放大器**。upper bound 的松紧随 chunk 长度增长(width 项 ~√(2 ln L),§2.2):指令区 chunk 64–256 token,bound 本来就松;再对多个 head 取 max,等于在"每个 head 一次松 bound 抽样"里挑最大——极值统计叠极值统计,排序被"最幸运 head 的最松 bound"主导而非真实相关性。DOM 元素块 6–64 token,envelope 紧(token 少则 M≈m,bound 贴近真实最大点积),max 传递的主要是信号。**同一个聚合函数,在紧 bound 上是信号放大器,在松 bound 上是噪声放大器**——这是"只给 DOM 用"的核心不对称性。
3. **失效机制对号入座**。§2.3 的机制 B(指令覆盖塌缩)病因是**覆盖量不足**,修复靠 floor(保量);失败复盘中不存在"某个关键指令 chunk 因 head-mean 稀释而落选"的案例。机制 A(needle 被挤出)的病因才是稀释/挤占——聚合函数的手术只应做在病灶上。
4. **外部证据同向**:LessIsMore(§2.2)在 reasoning 模型上实测"共识型跨 head 聚合 > per-head 特化",与我们 online 的 ④ 峰值聚合净负(46 vs 55,§4.5)方向一致——在中等 budget 下共识聚合是更稳的默认,head 特化的收益难以兑现。
5. **消融已补(2026-09-02,night7,23 dumps)——预测被证实**:`cds_allhm`(指令/rest 区改真 32-head-max 排序,DOM 不动)的 **sysCov 从 0.186 → 0.154(B1024,−17%)、0.272 → 0.231(B2048,−15%)**,needle 不变——head-max 在长块松 bound 上按"最幸运 head 的最松 bound"排序,选中的指令块捕获的真实 attention mass 反而更少,理由 2 的机制成立。同批 `cds_tmax`(DOM 区改真 32-head per-dim max)与部署版 sum-then-max 聚合在噪声带内打平(0.481 vs 0.465 / 0.685 vs 0.695):部署实现无需修改(且 ④ 整体已被 online 否证,此格仅作机制记录)。另一诚实注脚:② 族的 sysCov(0.186/0.272)远低于 fixed-16(0.343/0.499),但 online 的 none 却是 7 vs 31——floor 选出的高分指令块虽然总 mass 少、但足以维持格式遵从,再次说明 offline mass 类代理与 online 行为间的映射并不忠实(§6.2)。

### 4.2 Toy example:一次完整的打分与准入(数值为虚构演示)

**Context(带行号)**:

```
L1  <|im_start|>system
L2  你是浏览器 agent。输出 JSON:{"action":"click","index":N}
L3  规则:……(约 100 token 的操作规范)
L4  <|im_start|>user
L5  任务:搜索 "RTX 5090" 的价格
L6  历史:step1 已打开 shop.com 首页
L7  [12]<div 促销横幅:年度大促,全场八折!>
L8  [13]<a 首页>
L9  [54]<input 站内搜索框>
L10 [55]<button 搜索>
L11 (另有 8 个导航链接 [14]–[21],及 URL 收尾)
```

**Chunking(②④ 相同)**:指令区粗块 C1=L1–L3(80 token)、C2=L4–L6(30 token);DOM 元素块 E1=L7(12 token,div,不可交互)、E2=L8(6,a)、E3=L9(6,input)、E4=L10(6,button)、N1–N8=导航链接(各 6,a)。

**打分**:decode 时模型的 attention query 按 GQA 分成 4 组,每组 8 个 query head 平均成 q̄_g;对每个 chunk 用 envelope 算 mixmax_wn 分数,得每 chunk 4 个分数(虚构):

| chunk | g1 | g2 | g3 | g4 | head-mean | head-max |
|---|---|---|---|---|---|---|
| C1 规则块 | 1.4 | 1.4 | 1.4 | 1.4 | 1.40 | — |
| C2 任务块 | 1.5 | 1.5 | 1.5 | 1.5 | 1.50 | — |
| E1 横幅 div | 2.6 | 0.5 | 0.4 | 0.5 | 1.00 | 2.6 |
| E3 搜索框 input | 0.4 | 0.4 | 0.3 | **3.2** | 1.08 | **3.2** |
| N1–N8 导航 a | ~1.2 | ~1.2 | ~1.1 | ~1.1 | ~1.15 | ~1.2 |

g4 是检索型 head:任务说"搜索"→ 对搜索框强响应(3.2);g1 更多响应版式/醒目文本(横幅 2.6)。**head-mean 把 E3 的 3.2 稀释成 1.08**——低于横幅(1.00 附近)且低于所有导航链接(1.15),因为版式 head 在很多元素上都响应,而检索信号只在一组。

**准入对比(B=128,不计 always-include)**:

- **②(两区,全部 head-mean)**:sys-floor 32 token → C2(1.50,30)进,剩 2 退回全局。全局池 98 token 按 head-mean 排序:C1(1.40,80)进,剩 18;N1(1.2,6)、N2(1.18,6)、N3(1.15,6)进,剩 0——**E3(1.08)出局**。模型读不到搜索框那行,不可能输出 [54]。
- **④(三区 + 交互优先 + DOM head-max)**:sys 区 32 → C2 进;**DOM 区 70**(0.55×128)先装第一批(可交互):按 head-max 排序 E3(3.2,6)、N1–N8(~1.2,共 48)、E4(0.8,6)、E2(0.4,6)全部装入,共 66,剩 4;第二批(展示类)只有 E1(2.6,12),4 token 装不下,出局;全局区 26:C1(80)装不下跳过,E1(12)进。**E3 必进**。

toy 展示的正是 online 失败复盘中反复出现的模式:版式响应块(横幅/导航)与长指令块在 head-mean 全局排序下挤掉唯一带检索信号的目标元素;④ 的三个组件分别堵住三个漏洞(尺度混排、展示块抢预算、mean 稀释)。

### 4.3 offline 组件消融(同 setting:同 23 个失败富集 dump、同 budget,指标 = needle hit rate)

| 配置 | B1024 | B2048 |
|---|---|---|
| fixed-16 Block(最强 baseline) | 0.331 | 0.444 |
| ②(region-aware 原样) | 0.296 | 0.515 |
| ② + 三区划分 | 0.302 | 0.537 |
| ② + 三区 + 交互优先 | 0.399 | 0.679 |
| **④ = 再 + DOM head-max** | **0.465** | **0.695** |

同一 setting 下 ④ 比 ② 的 needle hit 高 **57%(B1024)/ 35%(B2048)**;三个组件贡献:划分 +2%,交互优先 +32%,head-max +17%(B1024 相对值)。

### 4.4 hierarchical selection 的否证(它为什么不在 ④ 里)

DOM 树上真正的 branch-and-bound 下降(envelope 的父节点分数支配子节点,可靠剪枝)offline 结果:needle 0.544 vs 平铺 ② 的 0.547,打分次数 **170 vs 112**——不更准且更贵。原因:web prompt 的树浅而宽,下降起点节点数已接近平铺 chunk 总数;打分 kernel 开销以 launch 为主,串行多轮下降增加延迟。结论:层级下降适用于 chunk 数上千的场景(≥50k token 完整 DOM 页面),当前 6–23k context 弃用;④ 用"预算三区划分"这种一步到位的结构先验替代逐层下降。

### 4.5 online 同 setting 状态矩阵(strat20,132 step)

| 配置 | B1024 | B2048 |
|---|---|---|
| full attention | 56(dense-equivalent 复测) | 57 |
| fixed-16 Quest | **55**(n=132,valid 100,none 31,0 超时) | 55 |
| ②(region-aware sf25) | **53**(补跑合并后 n=132 全清,valid **123**,none **8**) | 57 |
| ④(co-design) | **46**(n=128,none 26) | **52**(合并后;valid 104,none 9,17 步永久超时) |

**④@B2048 定稿(补跑合并后)**:干净步 115 上与 ② 配对 **13W/12L(52 vs 51,平)**、与 Block 配对 10W/8L(平)——表面 52 vs 57 的差距**全部**来自 17 个在 1800 s 内跑不完的步(两轮尝试均超时;逐 head DOM 打分叠在 torch 预算路径上,长 context 步 decode 过慢)。即 **④ 在 B2048 是吞吐失败而非准确率失败;在 B1024 是真准确率失败**(配对净 −6)。两个 budget 均无胜场 → ④ 整体 online 否证成立,且性质拆分清楚。

**B1024 终局判断(2026-09-02;② 的 3 步补跑与 ④@B2048 合并仍在队列,不改变方向)**:
1. **② 与 baseline 在 B1024 打平**:补跑合并后 **53 vs 55**(② n=132 全清、0 错误),同批干净步配对 13W/14L(净 −1,噪声)——**region-aware chunking 在 B1024 成立**(53 ≈ 55,n.s.),且行为侧写显著更好:valid 123 vs 100、none 8 vs 31(指令 floor 让格式遵从率高于 baseline 甚至 dense)——多出来的 valid 步是"选错元素的有效动作",B1024 下的损失模式已从"格式崩坏"转为"元素辨析"。
2. **④ 的赤字被干净归因到分配层**:② vs ④ 配对 **20W/14L**(125 共同步,净 +6)。chunking 相同、打分公式相同,差别只在三区硬分割 + 交互优先 + 峰值聚合——这三件事在 online 合计净负。机制:硬 floor 在普通步上强制花掉配额、峰值聚合与优先层在难例上的 needle 收益不足以抵消。**B1024 排序:baseline ≈ ② > ④**。
3. **④@B1024 输给 baseline 9 步**,而 offline needle 代理预测反向(④ 0.465 > baseline 0.331)——offline 代理第二次方向出错(§6.2):失败富集 dump 只在难例上排名,fixed-16 在普通步上的稳健性被系统性低估。
4. **fixed-16 从 B2048 → B1024 无损(55 → 55)**,B1024 仍在近饱和区;此前"≤2k 细粒度必需"的区间刻画(基于 ① 的 48 与伪 baseline)作废,改为:**B1024 下细粒度不是必需,但区域 floor 大幅改善格式遵从**。

---

## 5. Research Contributions

**可主张(按证据强度排序):**
1. **按 web-agent prompt 结构分区的 chunking 与 budget 分配**(元素粒度 DOM 区 / 粗粒度指令区 / 统一归一化排序 / budget floor)。已有 budget 分配工作在 head 维(Ada-KV)与层维(PyramidKV),**按 prompt 语义区域分配未见先例**。同 setting 证据:+9/+12 vs ①,B2048 达 full attention 水平,chunk 数 1/4.6,selection latency 低 1.4–1.75×。
2. **变长 chunk 上 upper bound scoring 的长度归一化**(只修 width term)。"变长 × upper bound"在文献中是空缺组合;对照证据:同一切分无归一化显著落后(89 vs 96,p=0.02)。
3. **action-space prior 进入 KV 选择**(交互元素优先)与**区域条件化 head 聚合**(DOM head-max):offline needle +40%/+57% vs 最强 baseline,但 **online @B1024 初步为负(46 vs baseline 55,§4.5)**——此条目前只能作为"离线机制发现 + online 否证的诚实记录"主张,除非补跑/②cell 揭示可修的分配层问题;不可作为正向贡献宣称。
4. **失效机制与适用边界的定量刻画**:双机制(A 粒度 × B 指令覆盖)及其分别被哪个组件治愈的闭合证据(§3.2 中间消融);budget 三区间(≥3k 饱和 / ≈3k 过渡 / ≤2k 细粒度必需);hierarchical selection 在浅宽树上的否证;offline 代理指标的正确使用条件。
5. **可复现工程件**:CUDA mixmax kernel(与 PyTorch 参考选择结果 Jaccard 0.99–1.00)、selection 时间复用的适用边界、两个上游 bug 修复。

**明确不属于本工作**:upper bound 公式(Quest)、head-mean 池化(TSA/业界既有)、语义树切分(TSA 既有)、selection 时间复用思想(LServe)。

**未闭合**:④ 的三个 online cell;跨 turn 的 selector 状态复用(KV 数据复用已拥挤——CacheBlend/EPIC/KVShare 等;selector 先验复用 + DOM 元素跨页面修改的身份保持未被占据);在线任务成功率评测。

---

## 6. 方法学与已修事故(evaluation integrity)

1. 配对子集 + sign test + bootstrap;低于 run-to-run variance 的差异不解读。
2. offline 代理指标只在失败富集的真实 query 流上用;最终裁决一律 online replay。**第二次方向性失误(2026-09-02)**:失败富集 dump 上的 needle 代理预测 ④ > baseline @B1024,online 实测 46 vs 55 相反——富集本身是偏差源(代理只在难例上排名,online agree 由全分布决定)。修正后的使用规则:offline 代理**只用于同族配置间的相对排序与机制诊断,不用于跨族(vs baseline)的胜负预测**;跨族结论一律等 online。
3. **失效 baseline 事故**:fixed-16 @B1024 首测用 flat 模式默认 max-chunk-size 256,chunk 数(~30)< top-k(64)→ 实际全选 = full attention;所得 56 与 full attention 一致,反而成为 harness 正确性的复测。修正腿(`--max-chunk-size 16`)已完成:**55**(0 超时)。教训:**每条 baseline 必须核对实际读取的 KV 量,不能只信配置名**。
4. 工程对策(固化):长时间运行后 GPU 内存回收慢 → 相邻实验静置 ≥420 s;错误中止阈值随实验长度缩放;远程轮询 ≥30 min(避免 sshd 封禁);NVML 故障走 docker restart。
5. **baseline 保真度声明**:本 harness 的 "Quest"/Block baseline 复现的是其**打分公式**(fixed-16 + upper bound / centroid),但 head 维度沿用全 harness 统一的"KV head 间取 mean + 全 head 共享选中集"约定;原版 Quest 是每 head 独立 top-k。内部对比因此在该轴公平;论文表述用 "Quest-style scoring" 并注明此差异。

## 7. 局限与后续

- 全部 agree 为 teacher-forced;发表前需在线任务成功率验证。
- ④ 在 B2048 的提升空间被 full attention(57)封顶,价值主要在 B≤1024,结论待 §4.5 矩阵补全。
- 后续方向(排序):跨 turn 的 DOM 元素身份复用(针对 prefill——web agent 每步重新 prefill 6–14k token 是延迟大头,逐字节前缀缓存只覆盖 ~59%);hierarchical selection 的超长页面场景;长度归一化在其他结构化输入上的普适性。

---

## 8. sglang 部署验证(2026-09-05 → 09-07)

### 8.1 环境:让真实 sglang 在 GB10 上跑起来

官方 sglang 尚不支持 DGX Spark 的 sm_121a(issue #11658 仍开放),三个环境障碍逐一解决后,`bsa_v2` 后端在真实 sglang 0.5.9 中完全初始化并服务:
1. 容器 `LD_LIBRARY_PATH` 指向系统 torch 2.10 的库,使 venv 的 torch 2.9.1 Python/C++ 版本混链 → 启动时把 venv 的 torch lib 置于最前;
2. 官方 sgl-kernel 0.3.21 的 aarch64 轮子不含 sm_121 机器码(`RMSNorm: no kernel image`),社区轮子又绑 torch 2.12 → 从上游 v0.5.9 源码把 gencode 收窄到 `sm_121a`、关 FA3 后自编(CUDA 13 把 CCCL 头文件移到 `include/cccl/`,须加入搜索路径);
3. BSA 选择 kernel 需按 venv torch 重编 → 加 `BSA2_KERNELS_DIR` 覆盖加载路径。

### 8.2 实现等价性:standalone vs sglang 是同一方法吗

固定同一 prompt(7515 token),导出两条路径首个 decode 步的选中集(`TSA_SELECTION_DUMP` / `BSA2_SELECTION_DUMP`),按 token 级重合度比较:

| 层 | chunk 起点 | 选中 token(standalone / sglang) | 共同 | token-Jaccard | 占较小集 |
|---|---|---|---|---|---|
| 0 | **149 / 149 全同** | 2063 / 2049 | 1985 | **0.933** | 96.9% |
| 16 | 同 | 2086 / 2043 | 2012 | **0.950** | 98.5% |
| 32 | 同 | 2076 / 2060 | 1798 | 0.769 | 87.3% |

浅层仅差**一个预算截止处的边界 chunk**,来源是已记录的取整规则差异(standalone "首次超过" 落在 2063–2086,sglang "最接近总量" 落在 2043–2060);深层发散是稀疏层数值漂移逐层累积(第 32 层的 query 取决于前 31 层稀疏 attention 的输出),不是逻辑错误——逻辑错误会在第 0 层就表现为整块系统性分歧。判定:**两条路径是同一方法的等价实现**,差异限于记账。

**此测试抓到的一个配置错位**:sglang 的树解析模式默认 `chatml`,standalone 用 `webarena`,对齐前 149 个 chunk 起点只有 7 个相同、token-Jaccard 仅 0.53–0.73。首轮 sglang 的 agree=55 因此作废;启动脚本已改为默认 `BSA2_TREE_MODE=webarena`。

### 8.3 准确率(正确几何,strat20 132 步,B2048,mixmax_wn,sys floor 0.25)

| 配置 | agree | valid | none | 超时 |
|---|---|---|---|---|
| **sglang · 本方法(webarena)** | **59** | 105 | 27 | 0 |
| sglang · dense 基线 | 57 | 98 | 34 | 0 |
| standalone · 本方法(rgv2f / ②) | 60 / 57 | | | |

配对:vs sglang dense 10W/8L,vs standalone rgv2f 11W/12L,vs standalone ② 11W/9L——全部在噪声带内。结合 §8.2,**方法在 sglang 中的实现正确,准确率与 standalone 一致**。大批量(并发 24、running 32)未破坏输出。

### 8.4 速度与利用率(同一 132 步)

| | 总耗时 | 每步 | GPU 利用率 |
|---|---|---|---|
| standalone · 本方法(并发 6,torch 路径) | ≈20,700 s | 157 s | ~12% |
| sglang · dense | **652 s** | 4.9 s | ~94% |
| sglang · 本方法(webarena) | 967 s | 7.3 s | ~90% |
| sglang · 本方法(chatml,105 chunk/prompt) | 939 s | 7.1 s | ~88% |

- 相对我们自己的 serve **快约 21×**,但这是 sglang 推理栈(连续批处理、并发 24、更优 kernel)的功劳,与方法无关;
- **同栈内本方法比 dense 慢 1.48×**。GPU 已被 dense 打满(约 94%),稀疏化省不下 attention 时间;而稀疏选择在 batch>1 时退回逐请求 Python 循环(`tree_sparse_selector.py:1069`,批量 13–23 时每步串行 23×48 次打分与准入),这部分开销成了纯增量。MoE 解码中 attention 本非大头(3B 激活参数的 FFN 才是),即便选择开销归零,端到端收益也有限;
- CUDA graph 实际几乎未生效:为避免内存超限,graph 最大 batch 限为 8,而实际批量 13–23 全走 eager。

### 8.5 结论与下一步

1. 在**已饱和的成熟推理栈**上,当前实现是**负加速**;要转正必须把预算选择做成批量 CUDA 路径,使选择开销随 batch 摊薄。
2. 方法的适用场景是 KV 读取占比更高的**更长 context**,或 GPU 未饱和的**小 batch** 服务——需要在这两个区间重新测量。
3. 两条路径的已知记账差异(页对齐保留区 1+2 页 vs token 窗口 4+128;取整规则)应在论文表述中注明;若需逐 bit 一致,统一到一种约定即可。
