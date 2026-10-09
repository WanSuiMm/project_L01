# L01 第三轮：BaseCompress 的点态界可否低成本升级为自适应鲁棒性？

**日期**：2026-10-10。**性质**：以 Chen et al. (2026) arXiv:2606.07205v1 的真实 Lemma 4.3、4.5、5.2、6.5、6.6 和 §7 为起点的独立推导草案。没有声称已独立复证原论文全部引理或证明新颖性。以下“定理 1”可完整直接证明；“定理 2”是带有清楚接口条件的组合命题；全原算法端到端结论仍须逐式移植验证。

## 0. 与第二轮相比的方向变化

上一轮构造的 Softmax 正权子集攻击，能否定“固定查询可靠 ⇒ 同种子自适应查询可靠”这一黑箱推理，但并未攻击 Chen 等人的具体 `BaseCompress`。本轮逐节阅读发现另一个值得优先审计的方向：**在原论文的 `d = polylog(n)` 参数区间，BaseCompress 的点态尾界、残差对 query 的确定性 Lipschitz 性、及 fresh randomness 的逐调用安排，可能足以把所有查询和自适应数据流的保证升级到相同的空间主指数。** 这与先前假设“自适应必然需要较大指数税”相反。关键工具是标准 net/union bound，而非新的概率不等式。

## 1. 原文真实接口（核对）

- Lemma 4.3: 对输入固定的 m 个 keys，以及**固定** q，输出半数子集，`|sum gamma(K,q)-2 sum gamma(C,q)| <= C H log(m/delta)`，其中 `H = sqrt(sup_U gamma(u,u) sup_V gamma(v,v))`。`BaseCompress` 使用随机性，时间 O(m²·T_kernel)。
- Lemma 4.5: binary merge-and-reduce（相同权）把固定查询误差转为流式保证。
- Lemma 5.2: 高温方案仅对正定残余核 `gamma_t(k,q)=exp(<k,q>)-sum_{l=0}^t <k,q>^l/l!` 使用 BaseCompress，低阶多项式部分由确定性矩 sketch 维护。
- Algorithm 2, Lemmas 6.5--6.6: 低温方案 `Partition -> recenter/rescale -> Compress -> weighted merge-and-reduce`，分块量级 `b ~ e^{r²(1+o(1))} n^{o(1)}/eps`；随机 Partition 的成功证书属于**对输入集本身的全局结构事件**。
- §7: 有带 `|w_i|<=1` 的 weighted-KDE BaseCompress (Lemma 7.4)，再还原多坐标 Attention 的 ||softmax||_2 \|V\|_F 误差指标。
- §3: 自回归自适应输入流被作者列为开放问题。

请注意：公开版本表述始终主要是 `for any fixed q with high probability`，不等于对所有 q 同时成立。

## 2. 定理 1：对真正的 BaseCompress 建立 simultaneous-query 高概率界

固定任意输入 K=(k_1,...,k_m)，`||k_i|| <= r`，`||q|| <= r`，取 `B=r²>0`。

### 2.1 完整 Softmax kernel

令输出子集 C，定义误差函数

`D_R(q) = sum_{i=1}^m exp(<k_i,q>) - 2 sum_{i in C} exp(<k_i,q>)`。

由于 `BaseCompress` 的输出是原输入子集，各元素系数 `sigma_i=1-2 1{i in C}` 属于 ±1（允许重复元素以索引计），因此**对任意随机输出**有

`||grad_q D_R(q)|| <= sum_i |sigma_i| ||k_i|| exp(<k_i,q>) <= m r e^B`。

令 `h=e^B`，`L=m r e^B`，选半径 `rho = min{r, h/L}=min{r,1/(m r)}` 的确定性 rho-net `N_rho`，满足

`|N_rho| <= (1+2r/rho)^d <= [3+2mB]^d`。

对该 net 上每个 q，以失败预算 `delta/|N_rho|` 使用原 Lemma 4.3 的点态界，union bound 可得：以概率至少 `1-delta`，net 每点误差 ≤ `C e^B log(m |N_rho|/delta)`。而所有连续 q 都在 net 某点 rho 邻域内，误差变化至多 `L rho <=e^B`。从而

**定理 1A**（完整 kernel，条件为 Lemma 4.3 中的隐式通用常数和任意 delta∈(0,1)）：

\[
 \Pr\left[\sup_{\|q\|\le r}|D_R(q)|\le
 C_1 e^{B}\left(1+\log\frac{m}{\delta}+d\log(3+2m B)\right)\right]\ge1-\delta.
\]

证明只使用原文点态定理、残差确定性 Lipschitz 界及固定 net。特别是，**自适应查询可以看到完整 coreset 后再挑 q**；同一“所有 q”事件仍然覆盖它。

### 2.2 高温 residual kernel（关键细节）

定义 

\[
 \gamma_t(k,q)=\sum_{\ell=t+1}^{\infty}\frac{\langle k,q\rangle^\ell}{\ell!},\quad
 H_t(B)=\sum_{\ell=t+1}^{\infty}\frac{B^\ell}{\ell!},\quad
 H_{t-1}(B)=\sum_{\ell=t}^{\infty}\frac{B^\ell}{\ell!}.
\]

`t>=0`；`t=0` 时 `H_{-1}(B)=e^B`。这个尾核正定，且 `gamma_t(k,k)<=H_t(B)`，故 Lemma 4.3 的点态尺度是 `C H_t(B) log(m/delta)`。

导数：

\[
 \|\nabla_q\gamma_t(k,q)\|
 \le r H_{t-1}(B),\quad
 \|\nabla_q D_{t,R}(q)\|\le m r H_{t-1}(B).
\]

取 net 半径

\[
 \rho_t=\min\{r,\,H_t(B)/(m r H_{t-1}(B))\}.
\]

并利用关键的精确简单比较：

\[
 H_{t-1}(B)=H_t(B)+B^t/t!,\qquad
 H_t(B)\ge B^{t+1}/(t+1)!,
\]

所以

\[
 B\frac{H_{t-1}(B)}{H_t(B)}\le B+t+1.
\]

因此 

\[
 |N_{\rho_t}|\le[3+2m(B+t+1)]^d.
\]

同样 union bound + Lipschitz 扩展得到：

**定理 1B**（截断尾核）

\[
 \boxed{\sup_{\|q\|\le r}|D_{t,R}(q)|
 \le C_2 H_t(B)\left[1+\log(m/\delta)+d\log(3+2m(B+t+1))\right]}
\]

以概率 ≥ `1-delta` 成立，对**输入固定但任意 query**同时成立。`|w_i|<=1` 的 weighted-KDE 版完全相同，因为系数绝对值最多 1。

这是一个完整的受限数学引理，不包含硬件可行性判断。

### 2.3 放大代价

当 `m<=n`、`d=(log n)^a`、`t=O(log n)`、`r²<=poly(n)`，上述 pointwise-to-uniform 的 log 因子比原 Lemma 4.3 至多多 `O(d log(n(B+t)))`，是 `polylog(n)`。该放大在特定参数范围内不改变以 n 或 1/eps 表示的空间复杂度**主指数**；但可能使实际常数/内存很差。

## 3. 定理 2：fresh randomness 下，逐调用 uniform strong coreset 可抵御自适应数据流

在一个自适应流协议中，第 t 个输入点可依赖之前全部公开输出、压缩结果、以往随机性（甚至内部已公开的摘要），但**不能依赖未来尚未抽取的随机性**。

把整个 Merge-and-Reduce / weighted-merge pipeline 的随机子调用按实际发生顺序 c=1,...,M 编号。给第 c 次调用新的独立随机数 `R_c`；其输入 `X_c` 在调用开始之前由历史 `F_{c^-}` 决定；每次调用满足“对该 X_c 所有 q 同时成立”的条件保证

\[
 \Pr(\mathcal G_c\mid\mathcal F_{c^-})\ge1-\delta_c.
\]

于是条件期望和 union bound 给出

\[
 \Pr(\cap_{c\le M}\mathcal G_c)\ge1-\sum_{c\le M}\delta_c
\]

只要调用次数有确定的多项式上限、每次设 `delta_c=delta/M_max`，就可在最终事件上对所有压缩调用同时应用**确定性的 coreset 误差组合**。由于每个事件覆盖所有 q，查询甚至可完全基于当前 coreset 和已观察误差自适应选择，不再发生固定查询误用。

本结果直接成立于由 BaseCompress (1A/1B)、全输入结构性成功的 Partition、以及严格按 fresh-coin 实现的加权分层 Merge-and-Reduce 组成的**抽象实现接口**。这不是对一条未知原始随机种子复用实现的保证。`R_c` 每次调用独立是本构造的要求，不是论文所有可能实现自动具备的性质。

## 4. 对原论文空间指数的条件性推论（未完成全算法 proof audit）

**高温**：原 §5 的空间项大致为

\[
 S_{\rm original}(t) \lesssim {d+t\choose t} +\frac{e^{r^2}}{\epsilon}H_t(r^2)\,\mathrm{polylog}(n).
\]

采用 1B 时，对每一个 `BaseCompress` 调用替换 `log(m/delta)` 为 `log(m/delta)+d log(3+2m(r²+t+1))`，并在所有时间 / 子调用之间分配失败预算。只要原引理误差合成中没有额外依赖固定 q 的随机事件，在原高温参数窗口 `r=log^{o(1)}n`、`log n <= d <= log^a n`、`epsilon∈(n^{-100},n^{-c})` 下，该增量为 `polylog(n)`，因而 **`(1/epsilon)^{1-1/a+o(1)}` 主指数有希望不变**。

**低温**：原 §6.5 的 weighted merge 约 `b = e^{r²(1+o(1))} n^{o(1)}/epsilon`，内存 `O(b log²(n/b))`。`Partition` 的输入集结构保证属于全查询陈述；对其中的 BaseCompress 子调用逐个使用 1A，增加最多 `O(d log(nr²))` 的相对误差预算因子。对于 `d=polylog(n)`、`r²=O(log n)` 之类窗口，这仍然是 polylog(n)，故低温主尺度可能同样不变。大 r 时若本来所需空间超过 n，直接完整存储是合法退路。

**从 KDE 到 Attention**：§7.1 Lemmas 7.3--7.6 使用带符号权重的 weighted-KDE、value 按数量级分桶及分母近似。1B 的 weighted 版本能覆盖每个固定的分桶输入，理论上再对 polylog(n) 个子摘要取 union。要完整宣称 attention 原始 `eps ||softmax||_2 ||V||_F` 的自适应流界，仍须逐项检查分桶重设、值的舍入、分母的稳定性、同时更新时序及多头情形；此处**不能直接把已证的 1B 宣称为全 Transformer 定理**。

## 5. 主要学术限制与碰撞

1. **证明工具相当标准。** `epsilon-net + union bound + Lipschitz` 是经典 strong coreset 生成法，Braverman et al. (NeurIPS 2021) 的 Theorem 1.1 早就建立了强 coreset 的 merge-and-reduce 自适应鲁棒组合原则。因此仅以定理 1 或定理 2 为核心的新颖性不足。
2. **同一流的注意力正确性 ≠ LLM 完整轨迹正确性。** 即使对实际自生成 KV 流 `K_t`、其任意适应性 query `q_t` 近似成立，也不代表它的 KV 流与原始全精度 LLM 的 `K_t^*` 相同。真实多层近似推理还需额外误差传播与稳定性/耦合分析。
3. **不同威胁模型。** 本定理抵御基于过去输出选择当前 key/value/query 的非预知对手；不抵御在调用发生前就知道该调用未来随机币的全知对手；也不自动保证全序列采样、动态 top-p 下的 TV。
4. **动态参数限制。** `d=polylog(n)`、有限 query 半径和至多 poly(n) 次 fresh-coin 子调用至关重要；当维数多项式增长、数值尺度无界或必须复用同一私有随机种子时，net 代价可能不再可忽略。
5. **不宜盲目宣传“解决了开放问题”。** 作者的开放问题可能意在更强模型/更精细参数依赖；且 2026 年的 Query-Oblivious Coresets 已独立给出强查询保证（其目标、权重与复杂度口径要逐项对照）。需要完整新颖性检索及证明审计后，才可决定是否形成短论文或技术注记。

## 6. 下一条值得攻的真问题

在上述 `BaseCompress` 升级有效的情况下，适应性本身在 `d=polylog(n)` 区间可能**不改变空间主指数**。因此继续攻击简单随机遗漏 coreset 的故事已不够。

研究重点应移动到 **self-generated KV perturbation / end-to-end Transformer fidelity**：实际近似模型在第 l 层的 KV 与同 token-prefix 下完整模型的 KV 不同，建立逐层的输入流稳定性

\[
 \|\operatorname{Attn}(\widehat K,\widehat V,\widehat q) - \operatorname{Attn}(K,V,q)\| \le \text{(upstream state perturbations)}+\text{(own compression error)}
\]

并证明在 `L` 层/`T` 步下的可用、非指数爆炸的增益界。严格分析不同层缓存的持久误差如何反馈至未来状态，并**把真实 memory 代回去**，这是比一般 `net lift` 更可能有研究新意的切口。


## 6A. 附加的 stream-to-stream 稳定性引理（初等、但给出了下一阶段正确接口）

单独固定两条同长度的 KV 流 `(K,V)`、`(K',V')` 与查询 `q,q'`，每个 key/query 的范数至多 `r`，每个 value 的范数至多 `Vmax`。假设逐对应条目：

`max_i ||k_i-k_i'|| <= kappa`，`max_i ||v_i-v_i'|| <= nu`，`||q-q'|| <= zeta`。

对 logits `s_i=<k_i,q>`，`s_i'=<k_i',q'>`，有 `||s-s'||_infty <= r(kappa+zeta)`。Softmax 的 Jacobian 沿任一插值路径满足 `||J_softmax(s) u||_1 = E_p |u-E_pu| <= ||u||_infty`（用均方差上界），因此

\[\|\mathrm{softmax}(s)-\mathrm{softmax}(s')\|_1\le r(\kappa+\zeta).\]

直接拆开 weighted mean：

\[oxed{\|\operatorname{Attn}(K',V',q')-\operatorname{Attn}(K,V,q)\|_2 \le 
u+V_{\max}r(\kappa+\zeta).}\]

如果一个 coreset 在实际、自适应生成的 `(K',V',q')` 上的当前误差是 `e_comp`，那么对同一 token 前缀的原始精确流，最终误差至多

\[e_{m comp}+
u+V_{\max}r(\kappa+\zeta).\]

这是 n-independent 的单层稳定性，但 `kappa,nu,zeta` 如何随 transformer 层和未来 token 自回归累积，仍然未知。特别是该引理要求两条同长度、逐条对应的 KV 流；不同 token 轨迹需要另外建立概率耦合。

---

## 7. 引用与待核步骤

- Chen et al., *Towards Tight Bounds for Streaming Attention*, arXiv:2606.07205v1, Lemmas 4.3, 4.5, 5.1--5.2, 6.5--6.6, 7.3--7.6, §3, https://arxiv.org/html/2606.07205v1
- Braverman et al., *Adversarial Robustness of Streaming Algorithms through Importance Sampling*, NeurIPS 2021, Theorem 1.1, https://arxiv.org/html/2106.14952v2
- Liberty, Andoni, Kleiner, *Query-Oblivious Coresets for Softmax Attention*, 2026, https://arxiv.org/html/2609.06327v2

**最终判定**：已推导可直接复核的 BaseCompress uniform-query lifting 定理，和一个基于 fresh coins 的适应性组合定理；原文高温/低温空间指数“可能不变”是待完成的全算法推论，不是已审核定理。现在不需要新 LLM 训练，也不应该开始大规模 benchmark。
