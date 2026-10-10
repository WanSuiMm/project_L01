# Round 7 查新对照与主张边界

日期：2026-10-10。检索是有限覆盖，不是原创性证明。以下以原始论文的指定版本为依据。

## 1. 本轮希望保留的命题

给定固定模型和固定 exact-prefill 前缀，各层独立做一次满足固定方向无条件 sub-Gaussian MGF 的 thinning；正权质量保持，后续 KV 由各模型自己生成并精确追加。在统一模型范数条件下，不预先构造全查询成功事件，也不对所有 token histories 做 union bound，即可控制每个压缩随机种子对应的**完整生成序列 KL**。

现有 GS-Thin 的 `nu=O(1/m)` 接入这个转移定理，得到 `v=C*h(n,T)/m²`。参考方向的期望 KL 不超过 v；压缩方向在 v<1/4 时不超过 `2v/(1-4v)`，并有逐种子高概率版本。

在等权、原记录子集选择这一既有类别中，一个整数网格障碍给出所有选择器的 `Omega(h/m²)` 下界。此下界没有扩展到任意正权摘要。

## 2. 最直接的近邻

### Low-Rank Thinning（2025）

https://arxiv.org/html/2502.12063v1

已核查 Definition 3、Section 4、Appendix B.6，尤其 Proposition B.6/B.7/B.9。它已给出 GS-Thin、KV-kernel 及 `nu=(2/sqrt(3))*sqrt(M)/m` 的无失败事件 sub-Gaussian 性。均匀抽样的有限总体方差和 `nu=O(m^-1/2)` 也已给出。

**因此不原创：**GS-Thin 算法、pointwise `m^-1` 误差阶、kernel embedding、相对于均匀采样的基础误差优势。

**本轮候选区别：**以整个自回归分布为损失、处理多层 off-target suffix KV、无全查询事件的概率转移。不是另一篇 thinning 算法论文。

### Express Language Modeling（2026，v2 于 10 月更新）

https://arxiv.org/html/2606.10944v2

已核查 Theorems 1–3 和 Theorem 3 后的解释。论文明确把其定理定位为 exact attention 所见输入流的 attention approximation；off-target multilayer approximation 和 sequential decoding 作为更困难的实证场景。它已有强 streaming 工程与理论算法，不能声称 streaming thinning 没有人做。

**本轮区别：**我们只压缩前缀一次、suffix 仍增长，算法协议比 Express 更窄；但证明目标为完整生成 KL，并把各模型自己生成的后续 KV 纳入理论。不是解决 Express 所有重复压缩问题。

### Nearly Optimal Attention Coresets（2026）

https://arxiv.org/html/2605.05602v1

已核查 Theorems 1–7。其主题是 bounded-query 球上的 attention coreset 上下界，研究原记录子集且允许 query/key/value 维度及范数影响复杂度。

**重要风险：**仅写“error ~ 1/m，因此 KL ~ 1/m²”并没有新意；等权子集的阶也与已有 uniform coreset 理论接近。Round 7 的卖点必须落在新的分布转移条件与对象上，而不是单独宣传 `m^-2`。

**不作额外断言：**本轮没有复证该论文所有 communication lower bounds，也没有宣称改善其维度、半径或信息论最优界。

### The risk of KV cache compression（2026）

https://arxiv.org/html/2607.01520v1

已核查 response profile、稀疏概率测度风险和 Theorem 5.4 的 causal reducer lifting。它确实考虑分母、随机 reducer、causal prefill 和 decoding，不得把这些内容说成空白。

**本轮区别：**使用完整 autoregressive law 的 KL；正向风险包含由 Q_R 改变路径分布的换测度步骤。等权原记录类不同于更广的稀疏正权概率测度类，不能相互冒充 minimax 下界。

### Polynomial Context-Truncation Sensitivity ...（2026）

https://arxiv.org/html/2605.25085v1

Section 3 明确把 query 当作完整模型给定的外生 side information，并说明 fully recursive layer-wise treatment 是扩展方向。

**本轮区别：**真实 Q_R query 和 suffix KV 会改变；reference query 只是 hybrid 证明中的比较对象，不是实际压缩 decoder 的 oracle 输入。

### Training Transformers for KV Cache Compressibility（2026）

https://arxiv.org/html/2605.05971v1

已有 prefix-cache replacement 协议和 histogram 类型构造。本轮不宣称发明该实验设置，也不由 histogram 下界构造推断 trained-model 的普遍压缩性质。

## 3. 经典概率工具：必须承认

- Hsu, Kakade, Zhang, *A tail inequality for quadratic forms of subgaussian random vectors*, 2012：二次型指数控制已有。
  https://arxiv.org/abs/1110.2842
- Russo and Zou, AISTATS 2016：信息论控制自适应选择偏差已有成熟研究。
  https://proceedings.mlr.press/v51/russo16.html
- 熵变分式由 tilted law 的 KL 非负直接推出；Hardy 不等式、KL chain rule、Pinsker 也是经典工具。

本轮最值得独立核查的是：把精确参考轨迹、次高斯随机摘要与完整生成 KL 的 self-bounding entropy closure 组合起来，是否已有同等命题。当前核查的近邻中未找到该完整陈述；这不是“全球没人做过”的证明。

## 4. 理论风险清单

1. **类别限制明显。** 等权 minimax 下界是整数网格精度障碍；允许两个自由正权原子即可破解其特例。它不能支撑“所有 KV 压缩的根本极限”标题。
2. **正向 KL 仅在小风险区间闭合。** 不能省略 `v<1/4`。参考方向和 TV 则不需要这个条件。
3. **模型常数仍然保守。** 没有解决先前 840 个 trained-model 比较全部 vacuous 的问题。
4. **原语依赖强。** 使用无条件 MGF。带失败事件的 Kernel Halving 需要额外处理坏事件；不能无修改套用本定理的 KL 界。
5. **随机化风险口径。** 下界是每种子生成 law 的 KL，再对种子平均；不适用于先把种子边际混合的 KL。
6. **构造时间。** 朴素 GS-Thin 是高阶多项式构造，不是已实现的快速推理方法。训练和 GPU 实验尚未运行。
7. **投稿判断。** 当前是具有明确定义、完整证明路线和近邻区别的候选增量结果；其论文级重要性仍未确定。

## 5. 与此前 toy 结果的关系

既有 20 个小模型和 histogram 结果可以作为数据基础，但不自动验证这个新 theorem。positive_cluster 不属于无偏 sub-Gaussian thinning 的自动实例；Uniform subset 属于，GS-Thin 需忠实实现原算法。

当前没有必要重训全部模型。后续实验应针对固定预算 thinning 与完整序列 KL，先复核 exact mathematical fixture，再复用已训练的 histogram checkpoint。不能通过某次 seed 的漂亮曲线宣称验证 minimax。
