# L01 Round 4 — Streaming-Attention 证明移植、Self-generated KV 稳定性与生成分布界

**日期：2026-10-10。目标：**继续上一轮指定的任务，逐项对齐 Chen et al., *Towards Tight Bounds for Streaming Attention*（arXiv:2606.07205v1）中的 Lemmas 5.2、6.5–6.6、7.3–7.6，并进一步给出多层 causal decoder 的明确、可核查的条件性定理。**不是证明原作者算法已经适用于通用现实 LLM，也不是声明获得论文级新颖性。**

## 0. 摘要与结论

1. **在原文的受限 `d=polylog(n)`、范数/位精度假设下，点态随机 coreset 界通过 net + uniformity + 每次 fresh coins 的升级，不改变原文上界的空间主指数。**高温区域可按 Lemma 5.2 的误差预算逐式完成；低温区域的 Lemma 6.5/6.6 需要保留 Partition 的结构性成功事件、原文的 `n^{o(1)}`，其统一化损耗是乘性 `polylog(n)`，而非免费。`§7` 中的带符号值分桶又增加 `polylog(n)` 的子摘要、失败预算；有限比特动态分桶需明示。没有复证原作者所依赖的全部外部 discrepancy 定理。
2. **新的可用接口结果：对于同一个 token 前缀，逐 token 的缓存误差不会因为序列长度 `T` 本身在层间递推中相加。**Softmax 对所有历史值作归一化的凸组合，以 `sup_{s≤t}` 的范数控制误差，因此误差在固定深度 L 下可写成一条只沿层传播的递推。
3. 组合得到**条件性闭环定理**：每次在线压缩调用以新随机数得到对当前数据所有查询有效的成功事件；若多层误差增益有界，则生成 `T` 步的分布偏差满足 `TV ≤ δ + W sqrt(T) E_L /(2τ)`。在坏事件后不假设仍准确；使用虚拟停止过程。
4. **难点没有消失，只是被明确定位。**随深度传播的增益乘积可能指数爆炸；现实 hidden/query 范数和 GPU 的空间—更新时间未审计；原算法 `BaseCompress` 的运行时间可能比完整 KV 更新贵。完整 theorem chain 依赖现有定理，方法新颖性弱。

---

## 1. 原论文核对 ledger：源文献到底给了什么

以 2026-06-05 的 v1 为固定引用版本（见文末链接），避免混用以后可能修订的公式。

| 原条目 | 源文献实际保证 | 本轮允许进行的升级 | 未覆盖项 |
|---|---|---|---|
| Lemma 4.3 | positive-definite kernel 的 `BaseCompress`，对 **固定 q**，半数子集，绝对核和误差 `O(H log(m/δ))` | 对 q 的确定性 Lipschitz + 有限 net，得到同一次随机压缩对全部 q 的成功事件 | 不改变数据、随机币的适应性威胁模型；不保证以后的内部隐藏态和精确 LLM 一致 |
| Lemma 4.5 | 二叉 Merge-and-Reduce，误差 `O(η(b)(j/b)log(j/b))`，空间 `O(b log(j/b))` | 各调用强查询事件 + fresh coins，按树确定性合成 | 树中每个调用必须确实使用新随机币 |
| Lemma 5.1 | 低阶多项式矩，确定性、所有 q 有效（含舍入） | 原样使用 | 位复杂度、范数与精度窗口 |
| Lemma 5.2 | `γ_t=exp_{>t}` 的流式残差 coreset；点态近似 `ε KDE` | `H_t [log(m/δ)+d log(m(B+t))]` 代入原 `b` 选择 | 只有 `d log(m(B+t))` 为多对数时，空间主指数才不变 |
| Lemmas 6.2、6.5 | Partition 给出结构性 pseudo-random / low-radius 分解；Compress 按输入相对 KDE 误差半缩 | Partition 的“全 q 结构”成功事件保留；每个重新居中/缩放的 BaseCompress 强查询化 | 复核所有路径概率、有限精度重心计算与准确性常数 |
| Lemma 6.6 | weighted Merge-and-Reduce；`b=e^{B(1+o(1))}n^{o(1)}/ε`；`O(b log²(n/b))` | uniform 子调用错误合成变成逐流路径有效 | 不得抹掉 `n^{o(1)}` 或处理成通用任意大 d 的保证 |
| Lemmas 7.3–7.6 | signed weighted KDE、value 坐标与幅度分桶、分母近似，归约到 normalized attention 指标 | 在**每个已确定的输入桶**使用 uniform weighted KDE + fresh coins，联合控制所有桶与坐标 | 分桶在线重设、值位宽、单层到所有层的因果传播仍须单独说明 |

原文链接：https://arxiv.org/html/2606.07205v1 。这些源论文结果不是本研究的原创引理。

## 2. 完整移植高温 Lemma 5.2：多出来的是什么

写 `B=r²`, `m=|K|`, `H_t(B)=∑_{j=t+1}^∞ B^j/j!`。固定此次 BaseCompress 的输入 `K`，但允许输出随机；令

\[
D_{t,R}(q)=\sum_{i=1}^m\sigma_i \gamma_t(k_i,q),\qquad
\gamma_t(k,q)=\sum_{j=t+1}^\infty \langle k,q\rangle^j/j!,\quad \sigma_i\in\{-1,+1\}.
\]

在整球 `||q||, ||k_i||≤r`，不论随机 R 的取值，都有

\[
\|\nabla_q D_{t,R}(q)\|\le mr H_{t-1}(B),\qquad
 B\frac{H_{t-1}(B)}{H_t(B)}\le B+t+1.
\]

取 `ρ= min{r,H_t(B)/(m r H_{t-1}(B))}` 的固定查询网，规模

\[
|\mathcal N|\le [3+2m(B+t+1)]^d.
\]

原 Lemma 4.3 用 `δ/|N|` 代替 `δ`，在整个网同时成功。沿 Lipschitz 延拓到球，得到

\[
\boxed{\sup_{\|q\|\le r}|D_{t,R}(q)|\le
C H_t(B)\{1+\log(m/\delta)+d\log[3+2m(B+t+1)]\}}
\]

以至少 `1−δ` 的概率成立。此命题对 **当前固定输入的所有 q** 同时成立，但不自动对一条事后窥见本次随机币而修改*这次输入*的 adversary 成立。

给记号 `F_t=C[1+log(m/δ)+d log(3+2m(B+t+1))]`。原 Lemma 4.5 对数层数记 `D=O(log(n/b))`，在同一次已发生流上合成的绝对误差是

\[
E_{\rm tail}(q)\le O\!\left(H_t(B)F_t\frac{j}{b} D\right).
\]

因为任意 q 有 `Z_j(q)=∑_{i≤j} e^{<k_i,q>}≥j e^{-B}`，选择

\[
\boxed{ b\ge C'\frac{e^B H_t(B)}{\epsilon}\,F_t D}
\]

即可得到 `E_tail(q)≤ε Z_j(q)` 同时对所有 q。严格地，`F_t` 用 `m≤n`、所有子调用数量 `M≤poly(n)` 以及失败预算 `δ/M` 的上界替代，避免 b 的隐式循环依赖。低阶矩的误差由 Lemma 5.1 确定性控制。

于是**条件性高温空间估计**：

\[
S_{\rm high}\le\operatorname{polylog}(n)\left[\binom{d+t}{t}+
\frac{e^B H_t(B)}{\epsilon}\{1+d\log[3+2n(B+t+1)]\}\right].
\]

在原文 `r=log^{o(1)}n`、`log n≤d≤(log n)^a`、`t=O(log n)`、`n^{-100}<ε<n^{-c}` 的窗口中新增项仍是 `polylog(n)`，选择原 t 后**形式上的主指数保留**：`S=\tilde O((1/ε)^{1-1/a+o(1)})`。这个结论是已有 Lemmas 4.3/4.5/5.1/5.2 的条件化强查询推论，不能宣称改善了原论文指数。

### 2.1 低温 Lemma 6.6 的容易忽略的循环

原证明逐层估计第 i 层 KDE，再用第 i 层的相对误差绑定第 i+1 层；直接在尚未完成证明前写 `Z_i(q)≤(1+ε)Z(q)` 有逻辑上的顺序问题。可以在抽象 merge forest 上用小增益递推代替：

\[
E_{i+1}(q)\le E_i(q)+a [Z(q)+E_i(q)],\quad a=C\alpha/b,
\]

其中 `α=e^{B(1+o(1))}n^{o(1)}·F` 归并了输入集结构与 uniform 基压缩的因素；所有块属于当前 disjoint stream stage，局部错误与当前质量相对成比例。归纳给出

\[
E_D(q)\le Z(q)[(1+a)^D-1]\le 2aD Z(q),\quad aD\le1/2.
\]

所以选 `b≥C α D/ε`。这一步要求真实 weighted merge 的中间对象按层形成相互不重叠的代表流，且每一块的相对误差调用成立；这个 invariant 正是源文献 Lemma 6.6 的结构假设，并非从 `(1+a)^D` 单独推出来。相较源论文原来的 `e^{B(1+o(1))}n^{o(1)}/ε`，乘上新的 `F=polylog(n)` 后仍保留 `n^{o(1)}` 级 overhead。因此在 **`d=polylog(n)` 与数值范围适当的交叉窗口**中，低温 KDE 主指数不变。

### 2.2 §7 从 KDE 到 Attention 的另外三个条件

1. **带符号权重:** 对 `|w_i|≤1`，`BaseCompress'` 残差仍是 `∑σ_i w_i e^{<k_i,q>}`，对 q 的确定性导数界不增大；可进行同样 uniform lifting。
2. **分桶:** 原 Lemma 7.6 把 value 的每个坐标按幅度分桶，近似 numerator 和 denominator。要在在线流实现时保持 `polylog` 个桶，需遵守原文有限精度/多项式幅度范围或等价的可逐步更新分桶；不能让桶根据未来标签或未来随机币选择之前的压缩币。
3. **归一化:** `|Dhat−D|≤ε D` 且 `ε<1` 可确保分母为正；一旦对当前输入所有 q 同时成立，ratio bound 也对所有 q 成立。各坐标、桶和所有调用的失败概率可由新随机币的条件性上界累加。

以上得到一个**条件性、单层流式 attention 压缩**结论：在这些明确假设下，原 `ε ||softmax||₂ ||V||_F` 尺度可扩展到任意*非预知*自适应 self-generated 输入流，且在特定 polylog-d regime 保持空间主指数。未复证所有原 lemma 中隐式常数、浮点实现与分桶重设，所以不把它写成对原仓库实现的无条件 theorem。

---

## 3. 多层 causal decoder：同 token 前缀的状态差不会自动按长度积累

令 `x_{0,s}` 为同一 prefix 中第 s 个 token 的确定性输入 embedding（包括位置编码）。第 ℓ 层在时间 s 的完整和近似 hidden 分别为 `x_{ℓ,s}, xhat_{ℓ,s}`；每一历史 token 的 KV 按当时的前层 hidden 计算并缓存。仅考虑 causal attention，不允许未来 token 反过来修改历史 token 的 exact hidden，且近似实现不可从缓存读取和改变其余同前缀以外的信息。

定义

\[
e_{\ell,t}=\max_{s\le t}\|\widehat x_{\ell,s}-x_{\ell,s}\|_2,
\qquad e_{0,t}=0.
\]

每层 Q/K/V 是线性映射，算子范数分别记 `w_{q,ℓ},w_{k,ℓ},w_{v,ℓ}`。假设完整/近似 key 与 query 均在半径 r_ℓ 的球内，value 范数 ≤V_ℓ。该层 attention 后的更新映射 `F_ℓ(x,a)` 满足：

\[
\|F_\ell(x,a)-F_\ell(x',a')\|
\le\beta_\ell\|x-x'\|+\sigma_\ell\|a-a'\|.
\]

这允许 LayerNorm、RMSNorm、MLP、residual 被包含于 Fℓ，只要在所分析域确实能给出有效的有限 Lipschitz 常数；不是所有训练好模型自动具有小常数。

### 引理 3.1：Softmax 对历史长度的稳定性

如果两条等长、逐 token 对应的 KV 流满足 `max_i ||k_i−khat_i||≤κ`、`max_i||v_i−vhat_i||≤ν`、`||q−qhat||≤ζ`，则

\[
\boxed{\|\operatorname{Attn}(\widehat K,\widehat V,\widehat q)-\operatorname{Attn}(K,V,q)\|\le\nu+V_\ell r_\ell(\kappa+\zeta).}
\]

证明：每个 softmax logit 扰动 ≤ `r(κ+ζ)`；对于 `p=softmax(s)`，Jacobian 满足 `||J_p u||₁=E_p|u−E_pu|≤||u||∞`，沿 logits 连接线积分；value 插值增量的凸组合范数≤ν。**没有显式历史长度 t 因子。**

### 定理 3.2：自生成 KV 的层间误差递推

假设第 ℓ 层在线压缩器在其**实际近似输入流**上的输出误差（对实际近似 q）满足

\[
\|\widehat a_{\ell,s}-\operatorname{Attn}(\widehat K_{\ell,\le s},\widehat V_{\ell,\le s},\widehat q_{\ell,s})\|\le\xi_\ell
\quad\forall s\le t
\]

则在相同 token 前缀下，令

\[
C_\ell=\beta_\ell+\sigma_\ell\bigl[w_{v,\ell}+V_\ell r_\ell(w_{k,\ell}+w_{q,\ell})\bigr],
\quad D_\ell=\sigma_\ell,
\]

有

\[
\boxed{e_{\ell,t}\le C_\ell e_{\ell-1,t}+D_\ell\xi_\ell}
\]

以及

\[
\boxed{e_{L,t}\le E_L:=\sum_{\ell=1}^L D_\ell\xi_\ell\prod_{j=\ell+1}^L C_j.}
\]

证明：对所有历史 s≤t，前层状态误差最多 e_{ℓ−1,t}；因此 key/query/value 逐点偏差依次最多 `w_k e,w_q e,w_v e`。用引理 3.1 和压缩误差三角不等式，再代入 Fℓ 的双输入 Lipschitz 界，取 `max_{s≤t}`，即得逐层递推。由 e_0=0 展开即得闭式。这里**没有在时间上再累加 `T` 次状态误差**；该保证只处理同一 token 前缀，不代表两条独立采样的 token 路径一定一致。

**必要警告：** `C_j` 的连乘可能随 L 指数增长。这不是证明技巧失败，而是没有层稳定性约束时无法普遍消除：单 token 的残差线性块 `x↦c x`、`c>1`，使输入误差按 c^L 放大。需要模型专属的 operator norm/局部增益证书，而不能把乘积视为常数。

### 3.3 把 normalized attention 指标代入 ξ

假设值有逐点范数界 `||v_i||≤V_ℓ`，keys/queries 均有范数 rℓ。由于 `max_i p_i≤e^{2r²}/n`，有

\[
\|p\|_2\|\widehat V\|_F
\le(e^{r_\ell^2}/\sqrt n)(\sqrt n V_\ell)=e^{r_\ell^2}V_\ell.
\]

故如果每层压缩器按 Chen et al. 的相对尺度返回 `εℓ||p||₂||V||_F` 的误差，可用

\[
\xi_\ell=\epsilon_\ell e^{r_\ell^2}V_\ell.
\]

这才使全层递推和真实 streaming KDE 的精度参数衔接；`e^{r²}` 可能非常大，不能藏在常数里。

---

## 4. 生成 T 个 token 的输出分布：可以直接组合出停止过程定理

假设对每个有固定长度的前缀，都有完全相同、确定性的完整 decoder 定义；其 logits 为 z_t；近似 decoder 使用 fresh-coin 压缩。输出抽样温度 τ>0、**不使用改变支撑集的动态 top-k/top-p**。词表输出头的算子范数 `W` 已知，若最后 hidden 误差≤E_L，则 logits 扰动振幅满足

\[
\frac{\operatorname{osc}(\widehat z_t-z_t)}{2\tau}\le b:=WE_L/\tau.
\]

令一次压缩调用在历史确定后有至多 δ_c 的失败条件概率，整个生成过程最多 M 次随机调用，`∑ δ_c≤δ`，每次新随机币在输入确定后才使用。定义首次失败前使用近似 logits、首次失败时起改用**精确 token 核**的虚拟过程 Q*；P 是原始精确模型生成分布。注意 Q* 仅用于证明，实际算法无需检测失败或运行 full KV。

- `Pr_Q(first bad)≤δ`，因为这是实际发生的自适应调用序列上的条件概率 union bound。Q* 在首次坏事件前具有相同历史，所以其坏事件概率相同。
- 在 Q* 中，坏事件之前的每步条件 KL 满足 `D_KL(q_t||p_t)≤b²/2`，坏事件后为零；连锁法则 `D_KL(Q*||P)≤Tb²/2`。（将独立随机币 pre-sample 为公用 tape 处理路径依赖。）
- 用 Pinsker 与停止耦合，

\[
\boxed{\operatorname{TV}(Q,P)\le\delta+\frac{W\sqrt T}{2\tau}E_L.}\tag{4.1}
\]

此处 Q 是**边际**生成序列分布（对压缩随机性平均），不是对每个随机种子都保证 TV。可以上界 1 作裁切。

这条引理比旧 Round 1 的 `Ψ(a,E)` 更简单，**因为假设更强**：不是仅对每个固定前缀平均 seed 成立的 pointwise 失败概率，而是在真实在线流程、fresh coins 条件下每个压缩调用有联合全查询成功事件；不会发生旧随机种子被查询反向筛选而放大失败概率的问题。

要达到 TV≤ρ，令 `δ<ρ`，充分选择

\[
\boxed{E_L\le\frac{2\tau(\rho-\delta)}{W\sqrt T}.}
\]

这给出真正的 `√T` 生成分布代价，而不是不受分析的训练 benchmark accuracy。

---

## 5. 精度分配、总空间与明确的非空参数区间

写 `g_ℓ=D_ℓ∏_{j>ℓ}C_j`，令层输出绝对压缩误差为 ξℓ。总稳定性预算约束

\[
\sum_\ell g_\ell\xi_\ell\le E,
\qquad E=\frac{2\tau(\rho-\delta)}{W\sqrt T}.
\]

若在指定参数 regime，各层压缩器的空间有 `S_ℓ(ξ_ℓ)≤A_ℓ ξ_ℓ^{-γ}`（同一个固定 γ>0；Aℓ 必须包含维度、范数、输入数值精度及 logarithms），则优化空间预算是一个初等但精确的 KKT 问题：

\[
\boxed{\xi_\ell^*=\frac{E}{H}(A_\ell/g_\ell)^{1/(\gamma+1)},\quad
H=\sum_j A_j^{1/(\gamma+1)}g_j^{\gamma/(\gamma+1)}}
\]

以及

\[
\boxed{S_{\rm total}^*=H^{\gamma+1}E^{-\gamma}.}
\]

这不是新的最优分配数学，它只是告诉我们不同层不应该机械使用统一 KV 精度；近输出且 downstream gain 大的层需要更细的压缩。

在所有 `A_ℓ,g_ℓ=O(1)` 且固定 L 的理想情况，`S_total=\widetilde O(T^{γ/2})`。对 Chen et al. 高温窗口 `γ=1−1/a+o(1)` 且 `T≈n`，得到**条件性** `S_total=\widetilde O(n^{(1-1/a)/2+o(1)})`。注意：原论文对 ε 与 n 的限制依然必须满足；`Aℓ` 中含 `e^{r²}Vℓ` 等范数因子，不能无条件当成常数；L、head 计数若增长，则 H 也增长。

**例：** `a=2`、`r=log^{o(1)}n`、`d≈(log n)^2`、`L` 固定，若增益、值范围和舍入因素均控制在 `polylog(n)`，则形式上是 `\widetilde O(n^{1/4+o(1)})` 缓存元素/相应位预算。它属于极端的受限理论 regime，**不代表实际 7B 模型满足这些范数界、推理更快或可直接替换 NVIDIA KV kernels**。同样，原 BaseCompress 的压缩操作时间可达 `O(m²)`，在实际服务系统中很可能比直接 append 一个 KV 更贵。

---

## 6. 论文资格与下一步裁决

**在当前证据下，L01 的简单“adaptive streaming KDE”版本很可能不够新的 NAACL 论文。**把原 Lemmas 的点态保证提升到全查询成功，再 fresh-coin 组合，以及用 standard stability + Pinsker 连接多层生成，都主要是经典组合技巧。Braverman et al. (2021) 已研究 adaptive robustness；Liberty、Andoni、Kleiner (2026) 的 *Nearly Optimal Attention Coresets* 研究 all-query 子集保证，Ofek I. Cohen (2026) 的 *Query-Oblivious Coresets for Softmax Attention* 进一步改善全查询构造。特别是后者报告 Qwen2.5-7B-Instruct 和 Llama-3-8B-Instruct 实际注意力头的几何参数 ρ≥23.877，其带有 e^ρ/ε 的 whole-ball 上界在这些头上会超过缓存本身；所以本报告没有真实 LLM 工作区间的非平凡空间证书。不能把这种归纳当作原创的近最优算法。源文献也没证明其理论高温 regime 覆盖实际 LLM 的工作点。

这轮得到的**有用成果**是：

- **结构上，序列长度不导致固定前缀 hidden-state 误差自动累加**；唯一一般必然的生成长度成本是 TV/KL 的概率组合（这里是 √T），而深度的 `∏Cℓ` 才可能爆炸。
- **原 upper bound 的 adaptive transfer 在 polylog-d regime 没有必然的指数障碍**；若要形成新的空间复杂度论文，需要找到超出标准 net upgrade 的 sharp tradeoff，例如 d 大时不能 uniformize 的必要额外代价，或者低成本控制 seed leakage 的新 primitive。
- **对于真正 NLP-native 结果，应研究可验证且非空的局部（layer/head-specific）增益条件、实际 coreset error budget，以及真实端到端时间/显存。**只用 worst-case 全局增益通常过于松弛。

建议在进入任何 LLM 训练/大规模推理前只做一次**小规模真实模型资格检查**：冻结一个 1B–8B decoder，对若干真实 token 前缀测量层间误差传递的经验局部放大、normalized error scale `||softmax||₂||V||_F` 与压缩更新时间；不进行新模型训练。若实际高温理论参数与经验增益无交集，就不要为了 NAACL 继续将经典 stability bound 包装成方法论文。

### 基础检查（已实际执行，非证明替代）

- 24 个 token × 4 层的因果 decoder 玩具实现：所有 `e_{ℓ,t}` 均低于定理 3.2 的递推界。
- 1,200 次随机两流 Attention 误差检查全部通过；观察到的最大 `实际/理论` 比值约 0.5321。
- 1,500 次 softmax KL 检查全部通过；观察到的最大 `实际/理论` 比值约 0.9849。
- **未**运行 Chen et al. 的 discrepancy coreset、**未**做实际 LLM、**未**对 streaming 时间/空间作测量；这些是分析工作与代数预检。

## 来源（固定阅读版本）

- Justin Y. Chen et al., *Towards Tight Bounds for Streaming Attention*, arXiv:2606.07205v1, §§4.2,5.1,6.1,7.1: https://arxiv.org/html/2606.07205v1
- Braverman et al., *Adversarial Robustness of Streaming Algorithms through Importance Sampling*, NeurIPS 2021: https://arxiv.org/html/2106.14952v2
- Liberty, Andoni, Kleiner, *Nearly Optimal Attention Coresets*, arXiv:2605.05602: https://arxiv.org/abs/2605.05602
- Ofek I. Cohen, *Query-Oblivious Coresets for Softmax Attention: Improved Bounds and Efficient Constructions*, arXiv:2609.06327v2: https://arxiv.org/html/2609.06327v2
- Alex Chen et al., *The Pitfalls of KV Cache Compression*, ACL 2026: https://aclanthology.org/2026.acl-long.1926/
- Laus et al., *KV Cache Compression Through the Lens of Transform Coding*, arXiv:2608.14191: https://arxiv.org/abs/2608.14191
