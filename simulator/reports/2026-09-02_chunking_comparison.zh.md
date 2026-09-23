# 同一 context 的五种切块对照:mixmax 分支(±region-aware)、main、CHUNKED_V2

**样本**:`2026-09-02_chunking_view_v2_cap32.md` 里的完整 browser-use prompt(11 062 token,Qwen3-VL tokenizer;重建后 token 总数与逐块计数和 V2 完全一致,对齐可信)。
**结构**:system prompt(~0–9327,含 DOM 格式示例与 JSON 输出示例)→ user turn(9328 起,agent 状态/历史)→ 真实 browser_state DOM(9608–11039)→ 收尾。
**生成文件**(同一渲染格式,可并排打开):`2026-09-02_chunking_view_mixmax_no_region_aware.md` / `2026-09-02_chunking_view_mixmax_region_aware.md` / `2026-09-02_chunking_view_main_subtree.md` / `2026-09-02_chunking_view_region_aware_on_main.md`;上游参考:`2026-09-02_chunking_view_v2_cap32.md`、`2026-09-02_chunking_view_subtree_full.md`。

---

## 1. 各机制到底怎么切(代码级)

### A. mixmax 分支·无 region-aware = `extract_leaf_chunks(tree, 16, 256)`
1. `parse_webarena_tree(token_texts)` 把 prompt 解析成树:ChatML 轮次为顶层,`[N]<tag ...>` 元素行成为叶子,其余文本按解析器的段落节点切;
2. 收集全部叶子,**叶子之间的缝隙**(父节点自己的属性 token)补成独立的 gap chunk;
3. 小于 16 token 的叶子与**下一个兄弟**合并(gap chunk 永不合并);大于 256 的硬切;
4. 全 prompt **一套参数**,没有位置概念。

### B. mixmax 分支·region-aware(c00f033,产出 57/53 的版本)
1. 同一棵树取叶子 span;用正则(`<|im_start|>user` 之后的 `\[\d+\]<`)定出**真实 DOM 区间**;
2. DOM 区间内:相邻叶子线性合并到 ≥6、硬切 ≤64(≈一行元素一块);区间外:合并到 ≥64、硬切 ≤256(粗块);
3. 切块直接对接 selection:mixmax_wn 打分 + TSA_BUDGET_TOKENS 准入 + TSA_SYS_FLOOR 保底——**切块与预算分配是一体设计**。

**开头三块的逐 token 对账(A 为何 52、B 为何 70)**:webarena parser 的第一个叶子就是 [3-54](52 tok,开头段落连着列表项 "1.";"2."–"6." 各自成叶)。A 中 52 ≥ min16 → 叶子原样成块(52 是**树叶尺寸**,不是合并产物);B 的指令区 min64 → 52+11("2.")=63 仍 <64,再吃 7("3.")→ 落块 [3-72]=70(**阈值越界点**)。另:token [0-2](`<|im_start|>system`)是父节点 token——A 的 gap 机制让它单独成块,B 无 gap 机制、这 3 个 token 不属于任何块(未覆盖缝隙的第一处;D 已修复)。

### C. main = `extract_subtree_chunks(tree, 16, 32)`(58b50e3,2026-09 设计)
1. 整个子树(标签+内容+闭合)≤32 token 就打包成一块;过大节点下钻,其开头/结尾残余 token **贴回本节的首/尾块**(标签不跨节);
2. 小块只在**同一父节点内**合并(不跨兄弟);保证精确平铺;全 prompt 一套参数。

### D. region-aware on main(本次移植,`extract_region_chunks`)
C 的子树提取按 (6, 64) 细跑一遍铺满全文 → 真实 DOM 区间外的相邻块再合并到 ≥64、≤256。= C 的边界质量 + B 的区域粒度。

### CHUNKED_V2(上游提案,纯切块原型)
**内容类型驱动**的三条规则,与树位置无关:DOM 元素与缩进子元素**向前绑定**打包(subtree packing);JSON 只在**括号平衡**边界切;纯文本按 16–32 token 在**行/句边界**打包。`SUBTREE_FULL`(401 块)是同一原型不带 cap32/JSON 规则的版本(文本区边界与 V2 逐 token 一致,同源)。

---

## 2. 总量统计(同一 11 062-token context)

| 版本 | 块数 | 中位 | p90 | 最大 | 真实 DOM 区块数 | 与 V2 起点重合 |
|---|---|---|---|---|---|---|
| **V2 cap32** | 446 | 27 | 32 | 32 | ~40(元素级) | — |
| SUBTREE_FULL(上游) | 401 | — | — | — | — | 文本区与 V2 逐点一致 |
| C main subtree(16/32) | 368 | 32 | 32 | 32 | 57 | 34/368 |
| **D region(new)** | 204 | 64 | 64 | 118 | 58 | 35/204 |
| A mixmax −RA(16/256) | 185 | **12** | 256 | 256 | 118* | 51/185 |
| B mixmax +RA(6/64+64/256) | 102 | 57 | 256 | 256 | 55 | 8/102 |

*A 的 118 个"DOM 区块"里大量是 1–11 token 的 gap 碎块;A 全文 <16 token 的碎块有 **103/185**(median 12 的来源)——gap 节点不参与合并是老实现的结构性缺陷。

---

## 3. 四个典型区域的边界实测(token 区间)

**system 纯文本区 [110–247](language_settings + input 列表)**
| | 块 | 边界 |
|---|---|---|
| V2 | 5 | [110-137] [138-167] [168-193] [194-216] [217-247] ← 恰在 `<language_settings>`、`<input>`、编号条目的行边界 |
| C main | 5 | [107-133] [134-165] … ← 同样 ~32 一块,但边界是树节点+32 步进,与 tag 段落错位 3–14 token |
| A −RA | 2 | [98-133] [134-316] ← 合并成 183-token 大块 |
| B +RA | 1 | [73-316] ← 244-token 粗块(设计使然:指令区只求覆盖) |

**system 里的 DOM 格式示例 [530–591]**
V2 按内容识别照样元素级(4 块,`[33]<div />`+子元素向前绑定);**B/D 把它当指令区粗块**(它在 user turn 之前)——这是"类型驱动 vs 位置驱动"的本质分歧:示例 DOM 永远不可点击,region 系有意不给它细粒度,V2 无此概念。

**真实 browser_state DOM 头部 [9608–9750]**
| | 块 | 特征 |
|---|---|---|
| V2 | 7 | 元素+子元素向前绑定,~10–32 tok |
| C main | 7 | 与 V2 几乎同构(32 上限的子树包)|
| D region | 3 | 子树包到 ≤64,数个小元素同块(如 `[727]<a …>` 一块、`<svg>+[752]<li>+…` 一块)|
| A −RA | 5 | 混着 [9651-9652] 这类 2-token gap 碎块 |
| B +RA | 3 | 线性合并的 ≤64 块,元素级但边界不如子树版干净 |

**JSON 输出示例区 [5489–5613]**
| | 块 | 特征 |
|---|---|---|
| V2 | 4 | **括号平衡**边界,每块是完整 JSON 片段 |
| C main | 5 | 32 硬步进,**切在 JSON 任意位置** |
| D region | 3 | 64 硬步进,同样不认 JSON |
| A −RA | 1 | 整个 256-token 大块 |

---

## 3.5 "语义程度"的量化:边界落在哪里

chunk 起点在所有 TSA 版本里都锚定在树节点边界(元素/章节不跨切);但节点超过 max 上限时,节点内部按固定步长切。两类边界的占比(本 context 实测):

| 版本 | 以换行结尾(语义行边界) | 正好等于 max(节点内部固定切) |
|---|---|---|
| V2 cap32 | 53.8% | 31.2%(内部切仍对齐句边界) |
| main subtree(max=32) | 15.5% | **82.1%** |
| mixmax −RA(max=256) | 50.8% | 16.8% |
| mixmax +RA | 63.7% | 31.4% |
| region new | 28.9% | 0% |

结论:main 的 max=32 在散文/JSON 密集输入上使多数 chunk 退化为"语义锚点 + 32-token 固定步长";老 mixmax 是粗粒度语义切(半数 chunk 即完整树节点);V2 是唯一对节点内部也做语义对齐(句边界/括号)的方案。

## 4. 结论

1. **A(老 mixmax 无 RA)最差**:gap 碎块(103 个 <16)与 256 大块并存,两头都不占——这正是 §2.3 机制 A(大块饿死 needle)的切块根源。
2. **B(老 RA)是"选择导向"的极简版**:块数最少(102,打分最便宜),真实 DOM 元素级、指令区最粗;边界质量最差(与 V2 起点重合仅 8/102),但它的价值本来就在与 budget/floor 的联动——online 57=dense/53≈baseline 是这套整体拿到的。
3. **C(main subtree)与 V2 在 DOM 区块数相同(7 vs 7)但切口质量不同**:V2 的切口全部落在行边界(每块=一个元素带缩进子行);main 在超过 32 token 的子树内部回到固定步长,切口会落在元素 id 中间(如 `[699]<header id=header_1-` ‖ `0 role=banner />`),且缩进空白会成为独立 2-token 小块(实测 [9651-9652]=`\t\t`,main 与 mixmax 都有此问题)。JSON 区同理:main 的 4 个切口全部落在键值对内部(把 `'type':` 切成两半),V2 的 3 个切口有 2 个落在逗号/右括号后。差距 = V2 的两条内容规则:**JSON 括号平衡**与**文本行/句对齐**。
4. **D(移植后的 region)= C 的边界 + B 的区域差异化**,块数 204 介于两者,继承 C 的平铺保证(无 gap 碎块、无跨节)。
5. **V2 与 TSA 系是正交的两层**:V2 是纯"切块质量"提案(内容类型驱动,无打分/预算概念);TSA 系的独有部分是 selection 侧(envelope 打分、budget、region floor)。若要合流,V2 可给 TSA 补两条规则(JSON 括号边界、文本行/句对齐——都是 `extract_subtree_chunks`/`extract_region_chunks` 里可加的局部规则);TSA 给 V2 补的是"切完之后怎么选"。
6. **三者(不含 region-aware)的取舍判断**:切块原则上 V2 最合理(原子=内容类型,块内同质性最高,内切也对齐句/括号);可运行实现上 main 最合理(原子=子树,无碎块、无跨节、平铺不变量、已接 serving);mixmax 的 leaf 方案是唯一被在线数据证明有害的(gap 碎块+大块稀释=机制 A),已被 subtree 取代。三者与打分的耦合:mixmax 的 max=256 换单元少(185)但必须靠 wn 归一化补救大小混杂;main 的 max=32 让块几乎等长(wn 失效、bound 紧)但散文区 82% 退化为固定内切;V2(446 单元)无打分成本约束。合理终点 = main 骨架 + V2 的 JSON 括号/句行对齐两条内部规则 + max 放宽到 48–64。
7. 一个待商榷的语义分歧:system prompt 里的**示例 DOM**,V2 给元素级、region 系给粗块。从 action-space 角度 region 系是对的(示例不可点击);从"模型要理解格式示例"角度 V2 有理。这属于可实验的开放问题,不是谁的 bug。
