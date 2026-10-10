# L01 Round 6：一次性前缀 KV 压缩的生成分布饱和界

日期：2026-10-10。

**性质：完整数学证明草案、针对性新颖性审计、可复算的数值证明检查。不是已发表定理，不是经过第三方复核的形式化证明，也不是训练后的语言模型实验。**

## 0. 核心结果与确切范围

研究一个固定深度、标准 dense causal softmax decoder。在长度为 n 的前缀完成精确 prefill 后，各层前缀 KV 仅压缩一次；后续 KV 在各自模型中正常生成、追加，不再压缩。

在下述统一范数、质量守恒和分子/分母近似条件下，完整生成序列的分布满足

\[
D_{\rm KL}(Q_{1:T}\Vert P_{1:T})\le C_{\rm model}\varepsilon^2\frac{nT}{n+T}.
\]

对应的 TV 上界是

\[
\operatorname{TV}(Q_{1:T},P_{1:T})\le
\min\{1,C'_{\rm model}\varepsilon\sqrt{nT/(n+T)}\}.
\]

存在实际的正权子集压缩和一层 softmax decoder，使这两个界在 n、T、epsilon 上分别达到同阶下界。因此，本报告得到的是**给定压缩精度的最坏情况传播率**，而不是“所有压缩器的最优空间复杂度”下界。

令 h(n,T)=nT/(n+T)。短生成 T<<n 时 h~T；长生成 T>>n 时 h~n。该转折不是要求每个模型的实测曲线都达到上界，而是统一上界及最坏情况匹配实例的尺度。

**范围没有省略：**精确追加使后续缓存仍需 O(T) 条记录。本结论不声称整个无限流可用固定或次线性内存维持。不保证反复压缩整个缓存时仍然饱和。不保证范数常数在实际大模型中足够小。它是 L01 中一个比“任意自适应流”窄、但保留标准 decoder 与自生成 KV 的问题。

## 1. 模型和时间索引

### 1.1 共同前缀下的两个 decoder

固定一个含 n>=1 个位置的原始前缀，先用完整模型计算每层原始前缀缓存

\[
\mathcal M_l^0=\{(k^0_{l,i},v^0_{l,i}):1\le i\le n\},\quad 1\le l\le L.
\]

随后将它替换成正权摘要

\[
\widehat{\mathcal M}_l^0=\{(\hat k^0_{l,j},\hat v^0_{l,j},w_{l,j}):1\le j\le m_l\},
\quad w_{l,j}\ge0,\quad\sum_jw_{l,j}=n.
\]

可用 log(w) 的 attention bias 实现正权；被保留 keys 的原有位置编码不能重新编号改变。

为消除 off-by-one 歧义，原始前缀之后先输入共同的起始标记 y_0。第 t 次预测先处理输入 y_{t-1}，将其 KV 追加到缓存，再预测 y_t。此时每层的逻辑总质量为 n+t。起始标记可理解为未纳入被压缩区域的 assistant/start 标记。其他标准 next-token 边界只改变一个索引；不要把包含当前位置和不包含当前位置的 KL 求和混用。

对每一个共同 token 历史 y_{<t}，分别运行原始模型与压缩模型，记第 l 层新位置 t 的隐藏状态为 x_l(t) 和 xhat_l(t)。二者输入 embedding 相同：

\[
e_0(t)=0,\qquad e_l(t)=\|\hat x_l(t)-x_l(t)\|_2.
\]

完整模型与近似模型的后续高层 KV **可以不同**：它们均从各自的 x_{l-1}(t) 生成。这里没有采用“后续 query/KV 冻结在原始模型轨迹上”的假设。

### 1.2 层的有效 Lipschitz 常数

为简洁先写单头情形。每层 query、key、value 映射（包括 attention 之前的归一化）具有统一 Lipschitz 常数 w_{q,l},w_{k,l},w_{v,l}。

全部原始、压缩、后续 query/key 范数 <=r_l，value 范数 <=V_l。设 B_l=r_l^2，attention 的 1/sqrt(d_k) 缩放吸收到 query/key 中。所有用于插值比较的向量也满足相同球约束。

attention 后的逐 token 映射 F_l 包含输出投影、残差和 MLP，满足

\[
\|F_l(x,a)-F_l(x',a')\|\le\beta_l\|x-x'\|+\sigma_l\|a-a'\|.
\]

最后 readout Psi（包括最终归一化）是 W-Lipschitz。输出分布为 softmax(Psi(x_L)/tau)，tau>0，不施加改变输出支撑集的 top-k/top-p。

所有常数必须对所声称覆盖的历史与时间统一有效，不能在证明最后把随 T 增长的量改称常数。固定数量的多头可逐头应用以下比较，再用输出投影的范数合成；本文件的显式常数仅对应单头。

### 1.3 前缀近似的是分子和分母

对全部 ||q||<=r_l，定义

\[
Z_l(q)=\sum_{i\le n}e^{q^Tk^0_{l,i}},\qquad
N_l(q)=\sum_{i\le n}e^{q^Tk^0_{l,i}}v^0_{l,i}.
\]

hat Z_l、hat N_l 使用摘要权重求和。假设

\[
|\hat Z_l(q)-Z_l(q)|\le n\epsilon_l^Z,\qquad
\|\hat N_l(q)-N_l(q)\|\le n\epsilon_l^N.
\tag{A}
\]

这是 unnormalized moment 条件；不能只用前缀 attention 比值误差代替。统一 epsilon 版本为 epsilon_l^Z<=epsilon、epsilon_l^N<=V_l epsilon。

## 2. 定理：显式的多层上界

定义

\[
\begin{aligned}
a_l&=\beta_l+2\sigma_lV_lr_lw_{q,l},\\
b_l&=\sigma_l e^{2B_l}(w_{v,l}+2V_lr_lw_{k,l}),\\
d_l&=\sigma_l e^{B_l}(\epsilon_l^N+V_l\epsilon_l^Z),\\
\mathcal E_L&=\sum_{l=1}^Ld_l\prod_{j=l+1}^L(a_j+2b_j).
\end{aligned}
\tag{B}
\]

**定理 1。** 在第 1 节的假设下，对每一个有限 T，

\[
\boxed{
D_{\rm KL}(Q_{1:T}\Vert P_{1:T})
\le\frac{W^2}{4\tau^2}\frac{nT}{n+T}\mathcal E_L^2.}
\tag{1}
\]

并且

\[
\boxed{
\operatorname{TV}(Q_{1:T},P_{1:T})
\le\min\left\{1,\frac{W}{2\sqrt2\tau}
\sqrt{\frac{nT}{n+T}}\mathcal E_L\right\}.}
\tag{2}
\]

这些是对完整自回归序列分布的结论，而非单条 teacher-forced 轨迹上的经验平均。若条件对所有相关历史都有效，相同推理也适用于反向 KL。

当 epsilon_l^Z<=epsilon、epsilon_l^N<=V_l epsilon，令

\[
H_L=\sum_l2\sigma_le^{B_l}V_l\prod_{j>l}(a_j+2b_j),
\]

则 E_L<=H_L epsilon，得到概要中的 C_model=W^2 H_L^2/(4 tau^2)。H_L 可能随深度指数增长；本定理不隐藏或消除这项成本。

## 3. 证明

### 3.1 旧前缀误差被后续质量稀释

固定第 l 层、时间 t、实际近似 query 和实际近似后缀 KV。先只把摘要前缀替换回完整原始前缀，保持 query、后缀完全相同。

设该混合缓存的 attention 输出为 A，摘要缓存输出为 Ahat。分子、分母差分别是 Delta N、Delta Z，则

\[
\hat A-A=\frac{\Delta N-A\Delta Z}{\hat Z+Z_{\rm suffix}}.
\]

因为权重非负、前缀质量 n、后缀 t 条单位质量记录，

\[
\hat Z+Z_{\rm suffix}\ge(n+t)e^{-B_l},\qquad\|A\|\le V_l.
\]

故

\[
\|\hat A-A\|\le e^{B_l}(\epsilon_l^N+V_l\epsilon_l^Z)\frac{n}{n+t}.
\tag{3}
\]

只要维持正权和质量，该推导不需要借助可能失效的“估计分母为正”近似，也不需要 epsilon 足够小。

### 3.2 对实际自生成后缀保留因果平均，而不是取最大值

接着比较两条都使用完整原始前缀的缓存：一条用近似后缀与近似 query，一条用完整模型后缀与 query。沿 key、value、query 的线性插值，令 A=sum_i p_i v_i。精确导数为

\[
\dot A=\sum_i p_i\dot v_i+\sum_i p_i(v_i-A)\dot s_i,
\quad \dot s_i=q^T\dot k_i+k_i^T\dot q.
\]

所有 content logits 落在 [-B_l,B_l]，所以比较中的每条原始记录有

\[
p_i\le\frac{e^{2B_l}}{n+t}.
\]

注意此比较已经换回了完整前缀，不能把某个大权重摘要原子错误地当成单位权重 token 套用这个不等式。

前缀 key/value 没有差；后缀第 s 个 token 满足

\[
\|\Delta k_s\|\le w_{k,l}e_{l-1}(s),\quad
\|\Delta v_s\|\le w_{v,l}e_{l-1}(s),\quad
\|\Delta q_t\|\le w_{q,l}e_{l-1}(t).
\]

又 ||v_i-A||<=2V_l，积分得

\[
\|\Delta A\|\le
2V_lr_lw_{q,l}e_{l-1}(t)
+\frac{e^{2B_l}(w_{v,l}+2V_lr_lw_{k,l})}{n+t}
\sum_{s=1}^t e_{l-1}(s).
\tag{4}
\]

将 (3)、(4) 代入 F_l 的 Lipschitz 界：

\[
\boxed{
e_l(t)\le a_l e_{l-1}(t)+b_l(H_ne_{l-1})(t)+d_l f_n(t),}
\tag{5}
\]

其中

\[
(H_n u)(t)=\frac1{n+t}\sum_{s=1}^t u(s),\qquad f_n(t)=\frac n{n+t}.
\]

这是路径上的确定性递推：固定任意一个完整 token 续写，全部历史状态是这条续写各前缀的一致状态；因果性确保未来不会重算或修改已缓存的历史 hidden。

### 3.3 一个自包含的离散 Hardy 界

对任意有限实数序列 u，

\[
\|H_nu\|_{\ell^2}\le2\|u\|_{\ell^2}.
\tag{6}
\]

证明：先考虑 C_{ts}=1{s<=t}/t。令 w_1=1，w_{t+1}=((2t-1)/(2t))w_t。归纳得到

\[
\sum_{s=1}^t w_s=2tw_{t+1},\qquad w_t/t=2(w_t-w_{t+1}).
\]

所以 sum_s C_ts w_s<=2w_t，sum_t C_ts w_t<=2w_s。对每一行使用带权 Cauchy–Schwarz 再求和，

\[
\begin{aligned}
\sum_t|(Cu)_t|^2
&\le\sum_t\left(\sum_sC_{ts}w_s\right)
\left(\sum_sC_{ts}|u_s|^2/w_s\right)\\
&\le 2\sum_s|u_s|^2/w_s\sum_tC_{ts}w_t
\le4\sum_s|u_s|^2.
\end{aligned}
\]

H_n 的非负矩阵逐项不大于 C，对 |u| 应用上式即可。此引理是经典 Hardy 不等式的证明，不作为 L01 原创数学工具。

另外，单调积分比较给出

\[
\frac13\frac{nT}{n+T}\le
S_{n,T}:=\sum_{t=1}^T\left(\frac n{n+t}\right)^2
\le\frac{nT}{n+T}.
\tag{7}
\]

上界由 integral_0^T n^2/(n+x)^2 dx 得到。下界由 integral_1^{T+1} 得到，且

\[
\frac{n^2T}{(n+1)(n+T+1)}
=\frac{nT}{n+T}\frac{n(n+T)}{(n+1)(n+T+1)}
\ge\frac13\frac{nT}{n+T}.
\]

### 3.4 累积状态误差有有限平方和

在一条固定共同 token 路径上，对 (5) 取 l2 范数，使用 (6)，

\[
\|e_l\|_{\ell^2(1:T)}\le
(a_l+2b_l)\|e_{l-1}\|_{\ell^2(1:T)}+d_l\sqrt{S_{n,T}}.
\]

由 e_0=0 递推，

\[
\sum_{t=1}^T e_L(t)^2\le S_{n,T}\mathcal E_L^2
\le\frac{nT}{n+T}\mathcal E_L^2.
\tag{8}
\]

这里不要求 a_l+2b_l<1。代价在深度乘积中明确保留，而时间上的递推是有界因果平均算子，不是另一个跨 token 的任意递归状态映射。

### 3.5 从共同前缀到真正的生成分布

令 z、zhat 为同一前缀上的两个 logits。log-sum-exp 的 Hessian 为 diag(p)-pp^T，其 Euclidean 算子范数至多 1/2。可用方差的范围界证明：对单位 u，Var_p(u_i)<=osc(u)^2/4<=1/2。

因此 Bregman 散度的二阶积分给出

\[
D_{\rm KL}(\mathrm{softmax}(\hat z/\tau)\Vert
\mathrm{softmax}(z/\tau))\le\frac{\|\hat z-z\|^2}{4\tau^2}
\le\frac{W^2e_L(t)^2}{4\tau^2}.
\tag{9}
\]

有限词表和 full softmax 确保条件分布具有共同正支撑。KL 连锁法则为

\[
D_{\rm KL}(Q_{1:T}\Vert P_{1:T})=
\mathbb E_{y\sim Q}\sum_{t=1}^T
D_{\rm KL}(Q_t(\cdot|y_{<t})\Vert P_t(\cdot|y_{<t})).
\]

(8) 对每条 y 都成立，因此可先求和再对 Q 取期望，得到 (1)。Pinsker 给出 (2)。这一步没有假设独立采样的 P、Q token 轨迹必须相同。证毕。

## 4. 正权子集构造给出匹配 KL 下界

令 n 为偶数，0<epsilon<=1。一层 scalar softmax attention 的前缀由 n/2 组配对记录构成：每组 keys 相同，values 为 +epsilon 和 -epsilon。

先取全部 keys、queries 为 0，使标准 softmax 为均匀注意力。完整前缀 numerator 为 0。压缩保留每组的 +epsilon 记录，赋权 2；总质量仍然是 n，denominator 精确，prefix numerator 改为 n epsilon。

等价地，也可以固定模型的前缀 value 为 +1、-1，只保留一正一负两个代表，赋予 n(1+epsilon)/2、n(1-epsilon)/2 权重；同样得到 numerator n epsilon。因此改变 epsilon 不必改变模型参数，且 n>=4 时这是严格减少 KV 条目的正权子集压缩。

后续新位置的 value 均为 0，keys、queries 仍为 0。输出 logits 设为 (a_t,0)，tau=1。这能通过选择固定 token embeddings、value projection 和 readout 嵌入一层 residual attention 模型；不需要把 attention 换成线性注意力。

于是

\[
P_t(1|y_{<t})=1/2,\qquad
Q_t(1|y_{<t})=\sigma(a_t),\qquad
 a_t=\epsilon\frac n{n+t}.
\tag{10}
\]

对 d(a)=KL(Ber(sigma(a))||Ber(1/2))，有 d(0)=0、d'(a)=a sigma'(a)。0<=a<=1 时

\[
\frac{\sigma'(1)}2 a^2\le d(a)\le\frac18a^2.
\]

由独立乘积分布的 KL 可加性与 (7)，

\[
\boxed{
\frac{\sigma'(1)}6\epsilon^2\frac{nT}{n+T}
\le D_{\rm KL}(Q_{1:T}\Vert P_{1:T})
\le\frac18\epsilon^2\frac{nT}{n+T}.}
\tag{11}
\]

这是一次合法压缩实际导致的风险，不是人为给最后 logits 加噪声。它只表明一般传播上界不能在 n、T、epsilon 上普遍改进；更聪明的压缩器在该特例上可能完全精确。因此 **不是 minimax 空间下界**。

### 4.1 非均匀、自适应 query 的加强

让不同配对组拥有不同有界 key；生成 token 可以决定下次 query 以及新写入 key，新 value 仍为 0。相同前缀中正负 values 成对抵消，完整模型仍然输出 Ber(1/2)，而近似 logits 是

\[
a_t=\epsilon\frac{Z_{\rm pre}(q_t)}{Z_{\rm pre}(q_t)+Z_{\rm new}(q_t)}.
\]

若 |q^Tk|<=B，则每条 token 路径上

\[
\epsilon e^{-2B}\frac n{n+t}\le a_t\le
\epsilon e^{2B}\frac n{n+t},\qquad a_t\le\epsilon.
\]

代入条件 KL 再用连锁法则，仍得 Theta_B(epsilon^2 nT/(n+T))。因此 KL 下界量级不依赖“查询永远为零”的特例。以下 TV 匹配证明仅使用 (10) 的独立子类即可。

## 5. 匹配 TV 下界（不只用 Pinsker 反向猜测）

把二元输出写成 X_t in {-1,+1}。P 是均匀 Rademacher 乘积分布，Q 的均值 m_t=tanh(a_t/2)>=0。

令 S=sum_t m_t^2。先证一般引理

\[
\operatorname{TV}(Q,P)\ge c_0\min\{1,\sqrt S\},
\qquad c_0=e^{-1/4}/16.
\tag{12}
\]

若 0<S<=1，设 theta_t=m_t/(2 sqrt S)，并选择有界检验函数

\[
f(X)=\sin\left(\sum_t\theta_t X_t\right).
\]

E_P f=0；另一方面

\[
\mathbb E_Q e^{i\sum_t\theta_tX_t}
=\prod_t(\cos\theta_t+i m_t\sin\theta_t).
\]

由于 0<=theta_t<=1/2，cos(theta)>=exp(-theta^2)、theta<=tan(theta)<=2theta。每个 x=m_t tan(theta_t) 在 [0,1)，故 x/2<=arctan x<=x。

乘积模长至少 exp(-sum theta_t^2)=e^-1/4；相位 A 满足 sqrt S/4<=A<=sqrt S<=1。因此 sin A>=A/2，得到 E_Q f>=e^-1/4 sqrt S/8。由 |E_Q f-E_P f|<=2 TV，得 (12) 在 S<=1 时成立。

若 S>=1，Hellinger affinity 满足

\[
\sum_x\sqrt{P(x)Q(x)}=
\prod_t\frac{\sqrt{1+m_t}+\sqrt{1-m_t}}2
\le e^{-S/8}.
\]

所以 TV>=1-affinity>=1-e^-1/8>c_0。(12) 证毕。

又对 0<=a<=1，tanh(1/2)*a<=tanh(a/2)<=a/2，因此

\[
\tanh^2(1/2)\epsilon^2S_{n,T}\le S\le\epsilon^2S_{n,T}/4.
\]

结合 (7)、(11)、(12)、Pinsker，得到绝对常数 c>0，使

\[
\boxed{
c\min\left\{1,\epsilon\sqrt{\frac{nT}{n+T}}\right\}
\le\operatorname{TV}(Q,P)
\le\min\left\{1,\frac\epsilon4\sqrt{\frac{nT}{n+T}}\right\}.}
\tag{13}
\]

可取 c=c_0*tanh(1/2)/sqrt(3)。这里严格证明了 TV 的匹配量级，没有把 KL 下界误认为 TV 下界。

## 6. 为什么不能把 (A) 换成前缀 attention 误差

取原缓存 (log3,0)、(-log3,0)，压缩为只保留 (log3,0)、权重 2。无论 query 是什么，两者 attention 都精确等于 0，总质量也都是 2。

追加一条 (0,1)，查询 q=1，则

\[
A_{\rm full}=3/13,\quad A_{\rm compact}=1/7,
\quad |A_{\rm full}-A_{\rm compact}|=8/91>0.
\]

因此“当前全部 query 上的 normalized attention 完全相同”也不是 append-closed 的保证。

更一般地，固定 query，原始与近似缓存的 A 完全相同而 Z 不同，追加权重 w>0、value u 的记录之后，差为

\[
\|u-A\|\frac{w|\hat Z-Z|}{(Z+w)(\hat Z+w)}.
\]

其对正权 w 的最大值在 w=sqrt(Z Zhat)，等于

\[
\|u-A\|\tanh\left(\frac14\left|\log\frac{\hat Z}{Z}\right|\right).
\]

这个归一化问题不是本报告单独的原创卖点：Haverbeck et al. 的 response profile 已含 denominator 坐标，Gelberg et al. 的 histogram 构造已揭示压缩改变前缀/后缀归一化比。此处作用是精确说明定理假设，不把已有认识包装成新发现。

## 7. 随机摘要、持续压缩与真实模型的边界

### 7.1 随机摘要

若初次压缩随机性 R 在 prefill 确定后抽取，并以至少 1-delta 的概率满足 (A) 的全查询事件，则好事件上对每个 R 都有定理 1。

边际输出分布 Q=E_R Q_R 满足

\[
\operatorname{TV}(Q,P)\le\delta+
\frac{W}{2\sqrt2\tau}\sqrt{\frac{nT}{n+T}}\mathcal E_L
\]

（右侧可截为 1）。这里一次性随机摘要不需要以后反复重抽随机币；全查询事件覆盖后续自适应查询。坏事件允许任意行为时，不由同一论证得到无条件的 KL 上界。

### 7.2 一般误差输入的能量形式

如果某种其他操作已独立证明层间递推

\[
e_l(t)\le a_le_{l-1}(t)+b_lH_ne_{l-1}(t)+u_l(t),
\]

则相同证明给出

\[
KL(Q\Vert P)\le\frac{W^2}{4\tau^2}
\left[\sum_l\|u_l\|_{\ell^2(1:T)}\prod_{j>l}(a_j+2b_j)\right]^2,
\]

前提是沿全部路径具有相应统一范数界。一次前缀压缩的 u_l=d_l f_n 有有限平方和；每一步持续引入常量误差的 u_l 则有 O(T) 的平方和。**不能把前者的饱和界直接移植到后者。**

### 7.3 不在本定理中的结论

本结果不包含：完整不断重压缩流的最优空间界；权重存储的位复杂度；GPU 加速；实际 LLM 一定满足小常数；非线性跨时间 recurrent toy 的稳定性；top-p 或有限窗口的自动推广；随长度增长的 attention logit range；最终任务准确率保证。

## 8. 已实际执行的检查

同目录 `verify_round6.py` 为完整可运行程序。结果在 `verification_results.json`，逐配置数据在 `scalar_rate_checks.csv`。

- 220 组解析下界实例检查，比较精确 KL 与上下界。
- 100 组独立二元乘积分布的完整序列 TV 枚举（T<=16）。
- 160 次 Hardy 数值检查和 5 个用于证明的 Schur 权恒等式检查。
- 4 个三层 causal residual decoder，每个穷举 256 条续写；检查每层递推、路径能量、序列 KL 连锁法则。后续高层 KV 确实不同，非共同缓存替身。
- 1 个非均匀 softmax、token-dependent query/new-key 的下界构造，穷举 1024 条续写。
- append 反例及正权 append 最优化公式的直接检查。

这些检查是证明/实现核验，不是正式模型训练或论文实验。没有用它们估计“新颖性概率”，也没有用有限随机检查代替证明。

## 9. 新颖性结论

数学结论与全球原创性是不同判断。本报告给出上述定理的完整证明；在 `NOVELTY_AUDIT.md` 所列原始来源及检索中，未发现同样的**一次性多层 prefix KV 摘要误差 -> 完整自回归 KL/TV -> nT/(n+T) 匹配尺度**结果。

最接近的已有工具包括 Hahn 的 soft-attention influence 衰减、Li et al. 的 Transformer 稳定性、Huan & Yuan 的 causal Hardy 界、Haverbeck et al. 的分子/归一化 response geometry。因此应将本成果称为**具有明确区别点的候选新主定理**，而不是宣称 Hardy、KL chain rule、注意力稀释或分母保持本身是原创。

仍需独立数学复核和更广引用网络检索后，才能确定其最终论文级新颖性强度。它已足以定义一个不依赖临时代理的理论实验问题。
