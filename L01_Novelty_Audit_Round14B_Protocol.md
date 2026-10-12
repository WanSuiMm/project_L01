# L01：Novelty Audit 与 Round 14B 终局协议

日期：2026-10-12。审阅版本：`WanSuiMm/project_L01@7f59ab2`。这是一份研究判断与新的**前瞻性**协议草案，不是已完成实验、审稿通过承诺或原 Round 14 结果修订。原 Round 14 的 `TASK_UNQUALIFIED` 与两个 Fisher transfer FAIL 不得被覆盖。

执行修订：以 [冻结协议](experiments/PROTOCOL_ROUND14B_SUFFIX_WRITE_V1.md) 与对应 JSON 配置为准。已明确一次性 Round 7 对照、cache-only 记忆消融、全部五种子的联合门槛与 Gate 4（终点至少一个策略 KL >=0.01 nat）。本草案不再是可直接运行的配置。

## 1. 当前证据边界

- Round 6：一次性前缀压缩，候选全生成分布 KL 传播上界 `C ε² nT/(n+T)`；全局范数数值证书无效。
- Round 7：用外部次高斯 thinning 原语，候选自生成 suffix KV 生成律 KL 转移；`h(n,T)/m²` 上界含巨大常数，正向 KL 需要小风险条件；限制为等权原始点子集的下界，不可推广任意加权摘要。
- Round 10–13：分析 Toy 中同样的局部压缩规模、选择次数及最终符号平衡，时间安排仍能改变完整生成 KL；Round 13 同样 Q 历史上的 Full Cross-event Fisher 能拟合 12.19x 差距，Diagonal 无法拟合。这是**离线输出几何解释**，不是新压缩算法、预测器、有限幅度严格证书。
- Round 14：八键生成式 associative recall 4/5 训练合格（seed 37 不合格）；全部 5 种子中 Fisher ≤25% 误差合格计数 Block 1/5、Alternating 3/5。仅保留四个合格种子进行探索性敏感性分析，平均 KL 分别约 1.72348e-4、9.95304e-5，Full Fisher 为 8.31012e-5、6.83489e-5，低估 51.8%、31.3%；不能通过剔除失败 seed 救回。原始 K/V Table 一直完整，压缩只发生在生成 suffix，记忆需求不足；根据当前 token ID 约定，`+` 在合法 query/answer 对中对应 answer、`-` 对应 query；局部误差相似性仅 1/5 通过。当前试验没有隔离 task transfer、扰动强度与语义角色。

## 2. 近邻文献与不可主张内容（检索截至 2026-10-12）

| 来源 | 已覆盖范围 | L01 边界 |
|---|---|---|
| Carrell et al., [Low-Rank Thinning, ICML 2025](https://proceedings.mlr.press/v267/carrell25a.html) | RKHS 子高斯 thinning、Attention-kernel、好的 `1/m` 误差原语 | thinning/核/速率不是本项目发明 |
| Han et al., [BalanceKV, NeurIPS 2025](https://arxiv.org/abs/2502.07861) | 在线/因果 Attention 的向量平衡与相关保留 | “平衡误差优于同向”本身不新；其端到端 LongBench 评估不等于自生成完整 KL 分布定理 |
| Gong et al., [Express Language Modeling, 2026-10-05 v2](https://arxiv.org/html/2606.10944v2) | Streaming causal Attention 保证与 GPU 工程；**Thm 3 明确保证给 exact-attention 输入流**，多层/解码 off-target 经验评价 | L01 Round 7 的 *per-seed 完整生成-law KL* 仍是可审计的区别，但需排除其推论已经覆盖同等模型 |
| Haverbeck et al., [The Risk of KV Cache Compression, 2026](https://arxiv.org/html/2607.01520v1) | Attention 响应 profile、query-dependent covariance、局部/全局 minimax 风险 | output-relevant covariance 不是原创；完整因果生成律与跨事件 Gram 是范围差异 |
| Gelberg et al., [Training Transformers for KV Cache Compressibility, 2026](https://arxiv.org/html/2605.05971v1) | 同一 histogram 函数可由不同可压缩性 Transformer 实现；KV-CAT | histogram 本身不是新 benchmark 或可压缩性发现 |
| Geh et al., [A Probabilistic Interpretation of KV Cache Eviction, 2026-08](https://arxiv.org/abs/2608.28293) | 驱逐 NP 困难、注意力 SNIS 视角与 decode-time correction | 压缩偏差与统计估计不新；未见相同完整序列跨事件 Fisher 定理 |
| Hwang et al., [Behavior-Preserving KV Cache Compression, 2026-10-05](https://arxiv.org/abs/2610.06479) | 以当前输出 KL 作为 eviction 评分并有长生成推断评估/加速 | “下游 KL 是最终指标”及 output-KL 打分不是新方法 |
| Chen et al., [The Pitfalls of KV Cache Compression, ACL 2026](https://aclanthology.org/2026.acl-long.1926/) | 多指令的顺序、驱逐偏差引发不同下游退化 | “顺序影响压缩质量”不是首创 |
| Arora et al., [Zoology, ICLR 2024](https://arxiv.org/abs/2312.04927) | MQAR 与 attention 的 recall 能力边界 | Round14 是自定义 generative adaptation，不得声称复现官方 MQAR |

**审计结论：**最有潜力的新对象是「明确区分 Attention 局部风险与自生成 suffix KV 下的完整生成律风险，以及有条件地保留随机 thinning 的 per-seed law guarantee」。最有力的实验证据是跨压缩事件因果响应的输出 Fisher *interaction*，但 Fisher 恒等式本身经典。尚未查到完全同型的组合定理，不等于已证明全球首创。不要将 Round 6–7 一次性前缀压缩与 Round 10–14 持续 suffix 压缩悄悄写成同一个已经证明的定理。

## 3. 可闭合的数学分离引理（范围明确）

固定偶数 `N` 个压缩事件，令旧前缀逻辑质量 `n=N`，所有 Attention content scores 相同。每次将一个值为 `(+a,-a)` 的两原子块以权重 2 保留其中一条，得到误差符号 `s_j ∈ {±1}`，在某个输出步骤上产生 logit 差

\[
\delta_j = \frac{2a}{N+2j}\,S_j,\qquad S_j = \sum_{i\le j}s_i.
\]

二元 softmax 的参考 logits 取 `(0,0)`，压缩 logits 取 `(\delta_j,0)`；`a` 为足够小的常数，保证所有 logits 在固定有界区间。由于条件概率不依赖此前采样内容，完整自回归分布的 KL 连锁法则给出

\[
D_{KL}(Q_{1:N}\|P_{1:N}) = \sum_{j=1}^N d\bigl(\sigma(\delta_j)\|1/2\bigr) = \Theta\bigl(a^2\sum_j S_j^2/(N+2j)^2\bigr).
\]

对 `Block=+^{N/2}-^{N/2}`，`∑S_j²=Θ(N³)`，所以 KL=`Θ(a²N)`；对 `Alternating=(+-)^{N/2}`，`∑S_j²=Θ(N)`，所以 KL=`Θ(a²/N)`。两种策略具有**完全相同**的正负次数、最终 `S_N=0`、每次压缩误差大小及对角二次项，但序列 KL **比值为 Θ(N²)**。注意这只是自回归 softmax 受限构造（position-only K/V 即可实现），不能当成一般多层 self-generated KV 定理。其思想与前缀 discrepancy 直接相近；可作为理论 separation，不足单独当 NAACL 主定理。

有限扰动 Fisher 可用下列标准但有用的精确桥梁描述：对每条固定历史，令 `δ=z(1)-z(0)`、`J=z'(0)`、`r=δ-J`，`p_s=softmax(z(0)+sδ)`，则

\[
KL(softmax(z(1))\|softmax(z(0))) =\int_0^1 s\;Var_{p_s}(\delta)\,ds.
\]

设 `D=max_i δ_i-min_i δ_i`，`a=√Var_{p_0}(J)`，`b=√Var_{p_0}(r)`，则在固定路径上有

\[
\tfrac12 c_-(D)(a-b)_+^2\le KL\le\tfrac12 c_+(D)(a+b)^2,
\quad c_\pm(D)=2\int_0^1 s e^{\pm sD}\,ds.
\]

理由：`e^{-sD}p_0(i)≤p_s(i)≤e^{sD}p_0(i)`，并用加权方差的最小二乘定义和 Fisher 半范数三角不等式。这个界清楚揭示有限幅度 Taylor 何时会失准：**输出 Fisher 范数中的余项 b/a 和 logit 振幅 D** 要小，而不是仅需局部 KV 误差小。它使用真实终点 `z(1)`，因此是 **a posteriori** 诊断而非 P-only 预先认证，也不是未经前作覆盖的新数学。

## 4. 若要继续一次：正式启动独立的 Round14B/15，而不是改写 Round14

研究目标不是“让实验通过”，而是“在**确实依赖生成后缀记忆**、且留存角色平衡的训练任务中检验完整跨事件 Fisher 是否优于对角近似”。

### 4.1 新数据生成法：Suffix-Write Delayed Recall（建议）

- 前缀只给指令/键字典，不给以后要检索的值。Suffix 反复产生 `WRITE, key_a, key_b, value_a, value_b`、延迟 filler 或其他写入、`READ, key, answer`。两个 `value` 独立随机，必须是在生成时才采样的**新的熵源**，不能从原前缀推出；后续 `READ` 应询问已发生且有非零间隔的 **latest write**，使旧前缀或立即邻居都不足以回答。
- 在写入事件里**只压缩相邻的同角色 `value_a, value_b`**，一条权重 2；正负选择表示第一/第二同角色原记录，不再表示较大/较小 token ID。每种策略用其反相版本；两者压缩事件数、时间和质量相同。确保压缩的是后续查询确实需要的记录。需要修改 `streaming_step` 的事件触发，使其基于写入结构而非简单所有偶数位置：这是**新任务/干预**，不能直接算作旧 Round 14 的复验。
- 如果新的模型无法学会长距 latest-write 检索，终止而不是扩大 sweep。训练阶段对随机价值/键的 top-1 不设不可能的 90% 门槛；分别测试其概率校准/NLL 和确定性 READ-answer 的正确率。整个 suffix 的全软最大熵随机 token 仍计入序列 KL，不将它们从风险估计中删除。

### 4.2 前瞻性 Gates（发布前冻结；供讨论的门槛，不是已验证的值）

1. **TASK**：预先确定 5 个训练种子，5 个全部报告。至少 4/5 模型对未见写入序列的延迟查询回答准确率 ≥90%，并且合法语法/查询率 ≥95%；若不满足，本轮评价为 TASK_UNQUALIFIED。若对模型更有把握可以继续冻结 all-five，但不能事后更改。
2. **MEMORY NECESSITY**：独立预注册的 causal ablation（删除/随机化生成 suffix 的写入值，不碰 query）应使答案准确率下降 ≥20 个百分点，或答案条件 KL 超过事先选择的非平凡阈值。若失败，说明仍是冗余后缀干预，停止。
3. **MATCHED INTERVENTION**：物理 `theta=1` 各策略记录数/权重相同；同角色保留；正负次数相同且双相。主实验每个合格 seed 的策略平均直接局部 attention-error² 最大/最小 ≤1.25 才能对“顺序而非局部失真”作强因果解读；未过仍要报告 KL，但降级主张。
4. **NONTRIVIAL RISK**：预冻结足够大的 KL 效应或 answer-effect 下界，并同时报告 Q 回答质量。若所有 KL 都约 1e-4 且答案完全不受压缩影响，不能用很小的绝对误差来伪装好的相对预测。
5. **FULL vs DIAG**：在 `theta=1` 物理缓存、Q 轨迹完整 KL(Q||P)、全事件 Fisher/Diagonal 使用**相同 T、方向、历史**，优先要求 Full 在合格种子里平均相对误差 ≤25% 且 Full 优于 Diagonal。`theta=.25` 只做局部有效性诊断，不是 equal-budget 物理压缩。5 种子独立，路径不是额外训练种子。
6. **BOUNDARY OUTCOME**：若 Full 在 theta=.25 成功、theta=1 失败且 logits/JVP 非线性余项大，视作**有理论意义的局部有效性边界**，而不是继续调整阈值或找好的随机种子。

### 4.3 Go/No-go：只允许一次新实验

- Gate 1/2 不通过：不是充分的迁移测试，停止扩展，保留 Round 14 负结果。
- Gate 1/2 通过，Gate 3 失败：压缩策略不可比，报告混杂，停止扩展。
- Gate 1–3 通过、Gate 5 通过：可称“在新任务中的定量输出几何迁移”，但仍是离线解释、非严格证书。然后**马上撰写论文**，不要再跑 Round 16。
- Gate 1–3 通过、Gate 5 失败：保留机制边界，理论集中在局部/有限偏差适用条件；不要再用额外种子或 θ 搜索“PASS”。

## 5. 论文定位与收尾

建议主线是 `From Local KV Distortion to Autoregressive Fidelity: Temporal Error Interaction and Its Limits`，而不是声称提出 SOTA 缓存压缩器。贡献候选：①完整生成律的限定转移与单次/连续误差传播；②匹配预算/局部误差的时序顺序分离与自回归 KL 的解析下界；③多层 trained toy 机制审计以及明确的迁移边界。不可声称通用可计算证书、一般真实 LLM 泛化、在线低成本预测或实际性能加速。

**发表新颖性仍是中等偏风险的候选**：BalanceKV 已有 discrepancy、Express 已有 causal Attention 保证、Haverbeck 有局部响应协方差、Geh 有概率驱逐与 decode bias。最可信的窄区别是 *full generative-law KL under independently evolving suffix KVs, per-seed random summary transfer*，及持续压缩事件交叉项的定量演示；它们还缺独立审稿级的前作等价性排查与更一般的实验证据。Fisher/Taylor、向量平衡、Histogram 和 KL chain rule 都不应写成原创。
