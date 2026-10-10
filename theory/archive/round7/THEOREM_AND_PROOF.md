# L01 Round 7：从次高斯前缀 thinning 到完整自回归分布

日期：2026-10-10。

**状态：本轮独立推导的证明草案，依赖明确列出的已发表 thinning 原语。未经过第三方或形式化证明核验。文献检索发现了明确的区别点，但不能证明全球原创性或论文录用价值。**

## 0. 结果、来源与边界

本轮给出两个结果。

1. 在一次性正权前缀压缩、随后精确追加各模型自己生成的 suffix KV 的协议下，把**点态次高斯摘要误差**转成完整生成分布的 KL/TV 保证。无须先建立一个全查询成功事件。对于压缩模型到完整模型方向的 KL，证明需要一个明确的小风险条件。
2. 在文献已有的**等权子集 thinning**类别内，给出生成 KL 的受限 minimax 量级：`h(n,T)/m²`，其中 `h(n,T)=nT/(n+T)`。反向 KL 对所有预算成立；正向 KL 的匹配上界在小风险区间成立。

上界算法 GS-Thin、其 `1/m` 次高斯参数和 KV kernel 都来自前作；并非本轮发明。新的候选内容是完整的多层自生成 KV 分布转移，以及指定压缩器类别中的联合 `(m,n,T)` 结果。

**不声称：**任意正权或任意编码器的 minimax 下界、重复全流压缩保证、实际 GPU 加速、解决先前实验中的巨大范数常数、全序列质量或所有 prompt 的统一证书。

`m` 是每层**旧前缀**保留的记录数，不是总缓存。后续缓存仍然需要 `T` 条记录。构造时必须先有完整 prefill；构造工作空间不计入压缩后的常驻前缀大小，但必须另外报告。

## 1. 协议与模型

沿用 Round 6 的协议。固定一个模型和长度 `n` 的原始前缀，先精确 prefill 各层 KV，再一次性替换它们。模型与原始前缀不依赖压缩随机性。使用独立的逐层随机种子 `R=(R_1,...,R_L)`；初次压缩后不再抽取压缩随机数。

每层摘要权重非负、总和为 `n`。新位置由完整模型 P 和压缩模型 Q_R 各自计算并以单位质量追加，故同一 token 历史上两者的后续高层 KV 可以不同。第 t 个预测前已追加 t 个新位置，含共同起始标记；逻辑质量为 `n+t`。

使用固定深度、单头、dense causal softmax decoder。多头版本需另行计入头数和输出投影。Q/K/V 映射包括 attention 前的归一化；Lipschitz 常数分别为 `wq_l,wk_l,wv_l`。query/key 范数至多 `r_l`，value 范数至多 `V_l>0`，`B_l=r_l²`。缩放 `1/sqrt(d_k)` 吸收到 Q/K。范数及 Lipschitz 常数对所有相关历史、插值点和时间统一成立。

attention 后的 token-wise 映射满足

\[
\|F_l(x,a)-F_l(x',a')\|\le\beta_l\|x-x'\|+\sigma_l\|a-a'\|.
\]

包含最终归一化的 readout 是 `W`-Lipschitz，生成用温度 `tau>0` 的完整有限词表 softmax。禁止动态 top-k/top-p 等支撑集变更。

定义

\[
 a_l=\beta_l+2\sigma_l V_l r_l wq_l,\qquad
 b_l=\sigma_l e^{2B_l}(wv_l+2V_l r_l wk_l),\qquad
 g_l=\prod_{j>l}(a_j+2b_j).
\tag{1}
\]

这些常数可能很大，且深度乘积没有被消除。

定义

\[
 f_n(t)=\frac n{n+t},\quad
 S_{n,T}=\sum_{t=1}^T f_n(t)^2,\quad
 h(n,T)=\frac{nT}{n+T}.
\]

单调积分比较给出 `h/3 <= S <= h`。

## 2. 次高斯摘要接口：已知算法提供，不是新的 coreset

使用 PSD kernel

\[
\kappa_l((k,v),(k',v'))
 =e^{k^\top k'}\left(1+\frac{v^\top v'}{V_l^2}\right).
\tag{2}
\]

它是已有 numerator/denominator KV-kernel 的范数重标定。令指数核 feature map 为 `phi`，满足 `<phi(k),phi(q)>=exp(k^Tq)`，于是

\[
\psi_l(k,v)=\phi(k)\otimes(1,v/V_l).
\]

令 `Delta_l` 是摘要与原前缀的**归一化 feature 均值差**：

\[
\Delta_l=\sum_j\frac{w_{lj}}n\psi_l(\hat k_{lj},\hat v_{lj})
 -\frac1n\sum_i\psi_l(k_{li},v_{li}).
\tag{3}
\]

假设对每个固定的 Hilbert-space 方向 `u`，都有无条件矩母函数保证

\[
\mathbb E_{R_l}e^{\langle u,\Delta_l\rangle}
 \le e^{\nu_l^2\|u\|^2/2}.
\tag{SG}
\]

这保证零均值及逐方向的次高斯尾，但**不等于同一个随机摘要对全部查询以某阈值同时成功**。不同层 Delta 独立。精确摘要 `nu_l=0` 的层从后续求和中删除。

[Carrell et al., Low-Rank Thinning, Definition 3 and Proposition B.6] 的 GS-Thin 在 `n/m` 为 2 的整数次幂时，选出恰好 m 个原记录并给每个记录相同归一化权重 `1/m`，满足

\[
\nu_l\le\frac2{\sqrt3}\frac{\sqrt{M_l}}m,
\qquad M_l=\max_i\kappa_l(x_i,x_i)\le2e^{B_l}.
\tag{4}
\]

原定义是有限 Gram 矩阵上的矩母函数式。Delta 位于输入 feature 的有限线性张成空间；把任意 u 投影到这个空间即可得到 (SG)，因此可以用于未包含在输入中的固定 query。

均匀不放回抽样也满足此接口，但仅有 `nu_l <= sqrt(M_l/m)`。确定性的任意聚类中心不自动满足 (SG)。先前训练实验中的 positive_cluster 不能直接套用本轮定理。

## 3. 主定理：完整生成分布转移

记 `d_{v,l}` 为 value 维数，定义

\[
 c_l=\sqrt2\,\sigma_lV_l e^{B_l},\quad
 s_l=\frac W{2\tau}g_lc_l,\quad
 q_l=\nu_l\sqrt{(d_{v,l}+1)e^{B_l}},\quad
 K=\sum_l s_lq_l,\quad
 \mathcal V=S_{n,T}K^2.
\tag{5}
\]

这里 `q_l` 是标量记号，不是 query 向量。所有量在生成之前由模型常数、预算和原语保证决定；不拟合观测 KL。

令 `D_R^- = KL(P || Q_R)`，`D_R^+ = KL(Q_R || P)`，其中 P、Q_R 都是完整长度 T 序列的分布。若 `mathcal V=0`，两者相同。否则：

### 定理 1A：参考分布方向，无小风险条件

\[
\boxed{\mathbb E_R D_R^-\le\mathcal V,\qquad
\mathbb E_R e^{D_R^-/(4\mathcal V)}\le e^{1/2}.}
\tag{6}
\]

所以令 `c_delta=1/2+log(1/delta)`，以至少 `1-delta` 的压缩随机种子概率有

\[
 D_R^-\le4\mathcal V c_\delta.
\tag{7}
\]

特别地

\[
\mathbb E_R\operatorname{TV}(P,Q_R)\le\sqrt{\mathcal V/2}.
\tag{8}
\]

对于混合分布 `Qbar=E_R Q_R`，凸性给出 `KL(P||Qbar)<=mathcal V`，继而同样的 TV 上界。每个种子先固定再生成的风险与混合分布风险必须区分。

### 定理 1B：压缩分布方向，显式小风险条件

如果 `mathcal V<1/4`，则

\[
\boxed{
\mathbb E_R\exp\left[\left(\frac1{4\mathcal V}-1\right)D_R^+\right]
\le e^{1/2}.}
\tag{9}
\]

因此

\[
\boxed{\mathbb E_R D_R^+\le\frac{2\mathcal V}{1-4\mathcal V}.}
\tag{10}
\]

以至少 `1-delta` 概率，

\[
\boxed{D_R^+\le\frac{4\mathcal V}{1-4\mathcal V}c_\delta.}
\tag{11}
\]

`mathcal V<=1/8` 时，期望不超过 `4 mathcal V`。小风险条件是这个证明的充分条件，**不是已经证明存在的真实相变阈值**。若要两个方向同时高概率成立，在 (7)、(11) 各用 delta/2。

## 4. 证明：改变 hybrid 比较的次序

### 4.1 把随机 forcing 放到参考 query，而不是实际自适应 query

固定任意 token 序列 y。P 的全部 query `q_l^P(t;y)`、suffix KV 和隐状态均由 y 确定，与压缩种子 R 独立。

在第 l 层比较两模型 attention 时，引入 hybrid：

- 原 Q_R：随机摘要前缀 + Q_R 自己的 suffix + Q_R query；
- hybrid：**同一随机摘要前缀** + P suffix + P query；
- 原 P：原始完整前缀 + P suffix + P query。

第一段中摘要前缀完全共同。它虽然随机，但保持正权、质量 n 和范数界。沿 Q/K/V 插值，suffix 单记录注意力概率至多 `e^(2B_l)/(n+t)`；前缀的大权重原子不被错误地按单位 token 上界处理。由 softmax 导数的凸组合形式，得到与 Round 6 相同的确定性传播项：

\[
 a_l e_{l-1}(t)+b_l(H_ne_{l-1})(t),\qquad
 (H_nu)(t)=\frac1{n+t}\sum_{s\le t}u(s).
\]

第二段中只替换前缀；query 为固定参考 query。令

\[
 z_l(t)=\frac{\widehat Z_l(q_l^P(t))-Z_l(q_l^P(t))}{n},\quad
 v_l(t)=\frac{\widehat N_l(q_l^P(t))-N_l(q_l^P(t))}{nV_l}.
\]

正分母下界 `(n+t)e^(-B_l)` 及 `||A||<=V_l` 给出 forcing

\[
 u_l(t)\le\sigma_l V_l e^{B_l} f_n(t)(|z_l(t)|+\|v_l(t)\|)
 \le c_l f_n(t)\|(z_l(t),v_l(t))\|.
\tag{12}
\]

于是路径上

\[
 e_l(t)\le a_l e_{l-1}(t)+b_l(H_ne_{l-1})(t)+u_l(t).
\tag{13}
\]

这一步没有把实际后续 KV 冻结成 P 的 KV；它们的差异已包含在第一段稳定性中。

### 4.2 有限秩评价算子和路径能量

对固定 y，定义线性算子 `T_l(y)` 把 Delta 映射为所有 t 上的

\[
 f_n(t)\big(\langle\Delta,\phi(q_l^P(t))\otimes e_c\rangle\big)_{c=0}^{d_{v,l}}.
\]

因为 `||phi(q)||²=e^(||q||²)<=e^B`，

\[
\|T_l(y)\|_{HS}^2\le(d_{v,l}+1)e^{B_l}S_{n,T}.
\tag{14}
\]

离散 Hardy 不等式 `||H_n||_(2->2)<=2` 和 (13) 给出

\[
\|e_L\|_2\le\sum_l g_l c_l\|T_l(y)\Delta_l\|.
\]

完整 softmax 的任一方向条件 KL 都不超过 `W²e_L(t)²/(4tau²)`。因此，**每条路径上，分别把任一方向的条件 KL 加起来**，均不超过

\[
 \left[\sum_l s_l\|T_l(y)\Delta_l\|\right]^2
 \le K\sum_l\frac{s_l}{q_l}\|T_l(y)\Delta_l\|^2
 =:U_R(y).
\tag{15}
\]

最后一步是带权 Cauchy–Schwarz。注意这里未声称两个 KL 方向的和也被同一 U 控制；每个方向分别成立。

令 `X_l=Delta_l/nu_l`，独立性使拼接的 X 在直和 Hilbert 空间中为 1-subGaussian。U 可以写成 `||A_y X||²`，各 block 为

\[
 A_{y,l}=\sqrt{Ks_l/q_l}\,\nu_l T_l(y).
\]

由 (14)，

\[
\operatorname{tr}(A_yA_y^*)
\le K S_{n,T}\sum_l s_lq_l=\mathcal V.
\tag{16}
\]

虽然 feature 空间可以无限维，Delta 只在有限前缀 feature 张成空间内，且 A_y 为有限秩；以下 Gaussian 积分等同于有限维线性代数。

### 4.3 经典次高斯二次型引理的自包含证明

若 X 为 1-subGaussian，G 为独立标准 Gaussian，则

\[
\begin{aligned}
\mathbb E_X e^{\lambda\|AX\|^2}
&=\mathbb E_G\mathbb E_X e^{\sqrt{2\lambda}\langle A^*G,X\rangle}\\
&\le\mathbb E_G e^{\lambda\|A^*G\|^2}
=\det(I-2\lambda AA^*)^{-1/2}.
\end{aligned}
\]

令 `lambda=1/(4 mathcal V)`。由 `||AA*||op<=tr(AA*)<=mathcal V` 与 `-log(1-x)<=2x` (`0<=x<=1/2`)，得到

\[
\boxed{\mathbb E_R e^{U_R(y)/(4\mathcal V)}\le e^{1/2}}
\quad\text{对每个固定 y 成立。}
\tag{17}
\]

同时 `E_R U_R(y)<=mathcal V`。这个二次型工具已有 Hsu–Kakade–Zhang 等经典前作，不作原创主张。

### 4.4 参考分布方向

KL 连锁法则及 (15) 给出

\[
 D_R^-\le\mathbb E_{Y\sim P} U_R(Y).
\]

P 与 R 独立，Fubini 得到期望界。Jensen、(17)、再 Fubini 得到

\[
\mathbb E_R e^{D_R^-/(4\mathcal V)}
\le\mathbb E_P\mathbb E_R e^{U_R(Y)/(4\mathcal V)}\le e^{1/2}.
\]

Markov、Pinsker、Jensen 分别给出 (7)、(8)。

### 4.5 压缩分布方向：用 KL 本身支付查询偏移，避免循环

同样有 `D_R^+<=E_(Q_R)U_R`。但 Q_R 与 R 相关，不能直接交换期望冒充独立。

对任意固定 R、lambda>0，熵变分不等式给出

\[
 \lambda\mathbb E_{Q_R}U_R
 \le D_R^+ +\log\mathbb E_P e^{\lambda U_R}.
\tag{18}
\]

可直接证明：对归一化 tilted law `P_lambda proportional P exp(lambda U_R)`，利用 `KL(Q_R||P_lambda)>=0` 即得。该换测度工具是经典的。

当 `lambda=1/(4 mathcal V)>1`，由 `D_R^+<=E_Q U_R` 得到

\[
 (\lambda-1)D_R^+\le\log\mathbb E_P e^{\lambda U_R}.
\]

指数化并对 R 求期望，用 (17)：

\[
 \mathbb E_R e^{(\lambda-1)D_R^+}
 \le\mathbb E_P\mathbb E_R e^{\lambda U_R}\le e^{1/2}.
\]

这正是 (9)。期望和高概率结论随之成立。**吸收之前明确要求 lambda>1，所以这里不是用待证结论当假设。**

## 5. 代入已有 GS-Thin：真正出现记录预算 m

由 (4)、(5)，

\[
\mathcal V\le\frac{C_*S_{n,T}}{m^2}\le\frac{C_*h(n,T)}{m^2},
\]

其中可取

\[
\boxed{
 C_* =\frac{4W^2}{3\tau^2}
 \left[\sum_l g_l\sigma_lV_l e^{2B_l}\sqrt{d_{v,l}+1}\right]^2.}
\tag{19}
\]

若把右侧记为 v，也可以在定理 1 中用更保守的 v 替代 mathcal V。

因此，对每个固定前缀：

\[
\mathbb E_R KL(P||Q_R)\le C_*h/m^2.
\tag{20}
\]

在 `C_*h/m²<=1/8` 时，

\[
\mathbb E_R KL(Q_R||P)\le4C_*h/m^2.
\tag{21}
\]

期望 TV 至多 `sqrt(C_*h/2)/m`，不要求小风险。

给定正向 KL 容忍度 eta 和种子失败概率 delta，充分选择

\[
 m\ge2\sqrt{C_*h(n,T)(1+c_\delta/\eta)}.
\tag{22}
\]

按允许的 dyadic 预算上取整；若超过 n，直接保留完整前缀。对固定模型常数和固定置信/误差，前缀预算是 `O(sqrt(h))`，而不是声明整个缓存是 `O(sqrt(T))`。

对均匀不放回采样，将 `nu=O(1/sqrt(m))` 代入同一证明，得到 `O(h/m)` 的 KL 上界（正向仍需相应小风险条件）。点态 `1/m` 对 `1/sqrt(m)` 的原语差别已是前作；本轮只研究其完整生成分布后果。

## 6. 等权 thinning 类内的下界：这次不是只展示一个坏算法

### 6.1 明确压缩器类别

令 `T_m` 为：只选择至多 m 个原始 KV 记录；保留原 key/value；每个保留记录赋相同权重 `n/s`（s 是实际保留数）；不修改 decoder、不改变新 KV 追加规则、不设置额外状态、不让 decoder 读取隐藏侧信道。选择可以依赖完整前缀和模型，可随机，可有无限计算时间。

这是文献 thinning 的固定预算选择类，不是任意正权/生成式 coreset 类。

### 6.2 合法一层 binary histogram decoder

使用一个固定模型，二元 token embedding 为 `(x,0)`，x in {-1,+1}。所有 query/key 为零；value 为 x；attention 输出 a 写入第二坐标；残差只保留第一个坐标；readout 输出 logits `(a,0)`。因而标准 softmax attention 为历史 token 的质量加权平均，输出为 Bernoulli(sigmoid(a))。

这是真正的自回归 token 模型：后续 value 由实际生成 token 决定，**不需要把 suffix value 人为设成零**。同一历史中两模型的新 value 相同（单层特例），但它们的序列分布不同。额外层可设为恒等映射，嵌入固定深度模型类。

取 `n` 是 `2m` 的整数倍。原前缀恰有 `n/(2m)` 个 +1，其他为 -1，前缀均值

\[
 \mu=\frac1m-1.
\]

任意 s<=m 的等权子集，若其中有 k 个 +1，其均值

\[
 \hat\mu=2k/s-1.
\]

若 k=0，`|hatmu-mu|=1/m`；若 k>=1，`2k/s>=2/m`，也有

\[
\boxed{|\hat\mu-\mu|\ge1/m.}
\tag{23}
\]

这个算术性质对每个随机种子成立，无论选择器多聪明。

设新输入 token 之和为 C_t（含共同起始 token），则同一历史上

\[
 a_t^P=\frac{n\mu+C_t}{n+t},\qquad
 a_t^{Q_R}=\frac{n\hat\mu+C_t}{n+t},\qquad
 a_t^{Q_R}-a_t^P=f_n(t)(\hat\mu-\mu).
\tag{24}
\]

两个 logits 均在 [-1,1]。Bernoulli log-partition 的二阶导是 sigmoid'，所以在此区间，两个 KL 方向分别满足

\[
 \frac{\sigma'(1)}2(a-b)^2
 \le KL(\operatorname{Ber}(\sigma(a))||\operatorname{Ber}(\sigma(b)))
 \le\frac18(a-b)^2.
\tag{25}
\]

将 (23)–(25) 沿任意共同历史求和，再应用各自方向的 KL 连锁法则：

\[
\boxed{
 KL(Q_R||P)\ge\frac{\sigma'(1)}6\frac{h(n,T)}{m^2},\qquad
 KL(P||Q_R)\ge\frac{\sigma'(1)}6\frac{h(n,T)}{m^2}.}
\tag{26}
\]

这对每个种子成立，因此也对 `E_R KL` 成立。

### 6.3 受限 minimax 的准确结论

设模型/前缀类 F 具有统一的 (19) 常数 `C_F`，并包含上述固定 histogram decoder 与允许的前缀。对于 `n/m` 为至少 2 的 2 的幂，定义

\[
 \mathfrak R_m^- =\inf_{\mathcal C\in T_m}\sup_{(P,\mathrm{prefix})\in F}
 \mathbb E_R KL(P||Q_{\mathcal C,R}),
\]

以及交换 KL 方向的 `mathfrak R_m^+`。GS-Thin 为上界算法，(26) 为所有选择器的下界，于是

\[
\boxed{c\,h/m^2\le\mathfrak R_m^-\le C_F\,h/m^2.}
\tag{27}
\]

在 `C_F h/m²<=1/8` 时，

\[
\boxed{c\,h/m^2\le\mathfrak R_m^+\le4C_F\,h/m^2.}
\tag{28}
\]

这是**指定模型类、指定等权压缩类、指定每种子 KL 风险**中的 matching rate。不能去掉这些限定语。

允许任意正权后，上述下界实例立刻失效：保留一个 +1 和一个 -1，分别赋予原两类的计数为权重，就能使全部前缀矩精确，完整后续生成也精确。允许存储任意新 value，甚至一个加权均值记录就够。随机种子边际混合 `KL(E_R Q_R||P)` 也不是 (26) 的风险对象。

## 7. 均匀抽样的独立比较下界

取 balanced prefix，n/2 个 +1、n/2 个 -1，同一 histogram decoder。均匀不放回抽 m 个记录，记子集均值 Z_R，则

\[
 E_R Z_R^2=\frac{n-m}{m(n-1)}.
\]

同一历史的 logit 差为 `f_n(t)Z_R`，由 (25)，

\[
 \frac{\sigma'(1)}2 S_{n,T}\frac{n-m}{m(n-1)}
 \le E_R KL(Q_R||P)
 \le\frac18 S_{n,T}\frac{n-m}{m(n-1)}.
\tag{29}
\]

参考方向也成立。m<=n/2 时为 `Theta(h/m)`。因此，在固定范数、小风险区间，已有 GS-Thin 和均匀抽样可有不同的生成 KL 预算阶；不是通过比较某个特意做差的任意算法得出。

## 8. 与已有结果的区别，以及尚未解决的事项

本轮不是把 `epsilon=1/m` 直接塞进 Round 6。Round 6 需要同一个摘要在整个 query ball 同时满足误差阈值；(SG) 只控制固定方向的概率，不能直接给那个成功事件。新的证明先把 forcing 移到精确模型的 query，建立逐参考路径的指数能量界，再用 KL 换测度吸收压缩模型自己产生的路径偏移。

但更大范围的原创性仍需独立审查。次高斯 thinning、Hardy 稳定性、entropy variational inequality、二次型指数界、等权网格量化障碍，都不是新工具。最值得审计的是它们是否已被组合成同样的多层、self-generated suffix、单次压缩、per-seed 完整生成 KL 定理。

没有解决：原实验中 `C_model` 巨大；任意正权压缩 minimax；重复压缩；低成本在线可计算证书；实际 trained-model 非空常数；数值舍入后的 SG 保障。

## 9. 引用

- Carrell, Gong, Shetty, Dwivedi, Mackey. *Low-Rank Thinning*. arXiv:2502.12063v1, Definition 3; Appendix B.6, Proposition B.6; also B.7–B.9 for construction runtime. https://arxiv.org/html/2502.12063v1
- Gong, Carrell, Dwivedi, Mackey. *Express Language Modeling*. arXiv:2606.10944v2, Theorem 3 and the paragraph immediately after it: exact-attention input stream is the theorem setting; off-target multilayer/decoding inputs are evaluated empirically. https://arxiv.org/html/2606.10944v2
- Haverbeck et al. *The risk of KV cache compression*. arXiv:2607.01520v1. https://arxiv.org/html/2607.01520v1
- Kim. *Polynomial Context-Truncation Sensitivity in Autoregressive Language Models: Sequential Wyner–Ziv Bounds for KV Cache Compression*. arXiv:2605.25085v1, Section 3. https://arxiv.org/html/2605.25085v1
- Gelberg et al. *Training Transformers for KV Cache Compressibility*. arXiv:2605.05971v1. https://arxiv.org/html/2605.05971v1
- Hsu, Kakade, Zhang. *A tail inequality for quadratic forms of subgaussian random vectors*. Electronic Communications in Probability 17, 2012. https://arxiv.org/abs/1110.2842
- Russo, Zou. *Controlling Bias in Adaptive Data Analysis Using Information Theory*. AISTATS 2016. https://proceedings.mlr.press/v51/russo16.html
- Project Round 6 proof, frozen at d53807c: https://github.com/WanSuiMm/project_L01/blob/d53807c/THEOREM_AND_PROOF.md
