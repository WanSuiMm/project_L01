# L01 Round 2 — Softmax witness leakage versus fully adaptive stream robustness

**Date:** 2026-10-09. **Status:** new self-contained proof drafts and executable finite sanity checks, not peer-reviewed or novelty-locked.  
**Aim:** develop an explicit softmax-attention counterexample for adaptive reuse of a randomized coreset; derive a constructive, but less space-efficient, positive result in the stronger adaptive-input protocol.

## 0. Do not confuse the two feedback protocols

(A) **Adaptive queries with frozen (K,V):** an adaptive observer gets one compressed attention answer and selects the next query.  
(B) **Adaptive input stream:** the observer also makes future key/value insertions depend on earlier compressed outputs. B strictly contains A if old data stay fixed until the second query.

Chen et al., *Towards Tight Bounds for Streaming Attention*, arXiv:2606.07205v1, Definition 2.2, Lemma 4.3 and §3 state fixed-query probability guarantees and flag adaptive input streams as an open direction. Lemma 4.2 already gives a polynomial-feature construction that supports deterministic uniform-in-query approximation at a potentially larger space cost.

This note does **not** attack or reimplement the particular randomized BaseCompress of Chen et al. The negative witness below uses a deliberately simple weighted deletion coreset and establishes **insufficiency of a pointwise guarantee as a black-box assumption**, not a space lower bound against all algorithms.

## 1. Concrete softmax counterexample: one tiny-error probe reveals the omission

Let n be an even, sufficiently large integer. Let u_1,...,u_n ∈ R^d be unit vectors, arranged in antipodal pairs (hence ∑u_i=0), such that for all i≠j,

    <u_i,u_j> ≤ ρ=1/20.

Such vectors exist in d=O(ρ^{-2} log n)=O(log n): sample n/2 independent random spherical unit vectors in sufficiently large dimension, apply the standard concentration and union bound to ensure |<u_i,u_j>|≤ρ for different base pairs, and add their negatives. Antipodal pairs have inner product −1 and meet the stated upper bound.

Fix B=20, r=√B. Let k_i=r u_i and v_i=u_i. For any ‖q‖≤r define

    A(q) = [Σ_i exp(<k_i,q>)v_i] / [Σ_i exp(<k_i,q>)].

Construct a **randomized weighted subset coreset** by choosing J uniformly from {1,...,n}, deleting (k_J,v_J), and assigning the same positive weight n/(n−1) to each survivor; the common weight cancels in normalized softmax, so its output is

    Â_J(q) = [Σ_{i≠J} exp(<k_i,q>)v_i] / [Σ_{i≠J} exp(<k_i,q>)].

Let s(q)=‖p(q)‖₂‖V‖_F be exactly the normalization in Chen et al.'s attention guarantee, where p is the *full* n-dimensional softmax vector and ‖V‖_F=√n. Set θ=3/4 and

    ε_n = 4 exp(θ B)/n,    δ_n = 1/n.

Assume n≥2 exp(2B) and choose n large enough that ε_n<1/2.

### Proposition 1: every predetermined query succeeds, but a two-query adaptive procedure fails with probability one

For every **fixed** q in the ball ‖q‖≤r,

    P_J{‖Â_J(q)−A(q)‖₂ > ε_n s(q)} ≤ 1/n.                         (1)

There is an adaptive two-query procedure whose first query q₀=0 is always accurate under this same ε_n, but whose second query

    q₁ = −r Â_J(0) / ‖Â_J(0)‖₂

is in the same ball and obeys

    P_J{‖Â_J(q₁)−A(q₁)‖₂ > ε_n s(q₁)} = 1.                       (2)

**Proof.** Write w_i(q)=exp(<k_i,q>), Z=Σ_i w_i. For any omitted J,

    Â_J(q)−A(q) = [w_J/(Z−w_J)] [A(q)−v_J].                     (3)

Because ∑k_i=0, Jensen gives Z(q)≥n for *every* q. Also w_J≤e^B and ‖A‖≤1, so for n≥2e^B,

    ‖Â_J(q)−A(q)‖ ≤ 4 e^{<k_J,q>}/n.                            (4)

By Cauchy-Schwarz, s(q)=√n‖p(q)‖₂≥1. Hence failure of the normalized threshold in (1) requires

    <u_J,q/r> > θ.

There is at most one such index J. Otherwise for two distinct unit vectors u_i,u_j and x=q/r with ‖x‖≤1,

    2θ < <u_i+u_j,x> ≤ ‖u_i+u_j‖,

implying <u_i,u_j> > 2θ²−1 =1/8 >ρ, contradiction. This proves (1).

At q₀=0, A(0)=0 and Â_J(0)=−u_J/(n−1). The normalized error is exactly 1/(n−1)<ε_n. The observed vector therefore reveals u_J, and q₁=r u_J.

At q₁, w_J=e^B and w_i≤e^{ρB} for i≠J. The weighted average Â_J(q₁) is a convex combination of {u_i:i≠J}, all with <u_J,u_i>≤ρ; therefore ‖u_J−Â_J(q₁)‖≥1−ρ. Rewriting (3) gives

    ‖Â_J(q₁)−A(q₁)‖ = [w_J/Z]‖u_J−Â_J(q₁)‖
      ≥ (1−ρ)e^B/[e^B+(n−1)e^{ρB}]
      ≥ (1−ρ)e^{(1−ρ)B}/(2n).                                 (5)

For the normalized metric, since ∑w_i²≤e^{2B}+(n−1)e^{2ρB}, Z≥n, and n≥e^{2B},

    s(q₁)=√n√(Σw_i²)/Z ≤√(1+e^{2ρB})≤2e^{ρB}.               (6)

Consequently the normalized error is at least

    (1−ρ)e^{(1−2ρ)B}/(4n).

This exceeds ε_n=4e^{θB}/n because

    exp[(1−2ρ−θ)B]=exp(3)>16/(1−ρ)=320/19.

This proves (2). ∎

**Important limits.** The chosen coreset omits only one of n distinct entries, so the witness is *not* a competitive compression scheme, not a lower bound on memory, and not an attack on Chen et al.'s algorithm. The required asymptotic n can be enormous because proof constants were chosen for simplicity. It is nonetheless a genuine positive-weight softmax subset, with bounded key/query/value norms, d=O(log n), and the exact normalized error metric of the cited paper.

**Optional sublinear-cache variant (illustrative, not harder):** repeat each of the n distinct key-value pairs L=n times, so the original stream has N=n² records. Store one representative of each surviving distinct pair with common weight proportional to L, and randomly omit one *type*. The full attention, coreset attention and normalized accuracy scale are unchanged because ‖p_full‖₂=‖p_type‖₂/√L and ‖V_full‖_F=√L‖V_type‖_F. The coreset uses n−1=Θ(√N) representatives; the same attack works. But exact deduplication already yields Θ(√N) storage, so this is not evidence of an intrinsic space/adaptivity tradeoff.

## 2. The underlying variable is seed information leakage, not error magnitude

Let R be the random coreset seed and H be a feedback observation. Suppose every fixed q satisfies Pr_R[B(R,q)]≤δ, and the posterior has a max-density-ratio bound

    dP(R|H=h)/dP(R) ≤ exp(I)      for P_H-almost every h.

Then for any query chosen as q=f(H),

    Pr[B(R,f(H))] ≤ exp(I) δ.                                    (7)

This follows by conditioning on h, applying the posterior density-ratio bound, and integrating.

In Proposition 1, H=Â_J(0) uniquely identifies J, so the posterior/prior likelihood ratio is n. Thus I=log n, and (7) gives n·(1/n)=1, exactly matching (2). The *norm* of the first probe error is only O(1/n); its *direction* reveals log n bits. Suppressing error magnitude alone does not control seed leakage. This is a standard max-information/measure-change principle, not a novel general inequality.

The argument assumes sufficiently precise observation of H to distinguish n outputs. With coarse quantization or injected noise, the attack's information channel can change. Any new theorem that relies on quantization must explicitly account for finite precision.

## 3. Constructive positive result: deterministic Taylor moments handle arbitrary adaptive streams

Let all keys and queries lie in the radius-r ball, B=r², and let p_t(x)=Σ_{l=0}^t x^l/l!. For all |x|≤B, Taylor's remainder obeys

    |p_t(x)−e^x| / e^x ≤ e^{2B} B^{t+1}/(t+1)! =:η_t.           (8)

Suppose η_t<1. For each multi-index α∈N^d with |α|≤t, maintain the **deterministic streaming moments**

    M^0_α = Σ_i k_i^α/α!,    M^v_α=Σ_i (k_i^α/α!) v_i.

At each query q, evaluate

    Z_t(q)=Σ_{|α|≤t}q^α M^0_α,
    N_t(q)=Σ_{|α|≤t}q^α M^v_α,
    Â_t(q)=N_t(q)/Z_t(q).

By the multinomial theorem, Z_t=Σ_i p_t(<k_i,q>) and N_t=Σ_i p_t(<k_i,q>)v_i.

### Proposition 2: fully adaptive module-level attention accuracy

For *any* sequence of keys, values and queries, including ones chosen in response to all previously published sketch outputs, at every step

    ‖Â_t(q)−A(q)‖₂ ≤ [2η_t/(1−η_t)] ‖p(q)‖₂‖V‖_F.            (9)

Moreover, the memory consists of (d_v+1)·binom(d+t,t) scalar moment coefficients (plus bookkeeping), and the per-item update and per-query evaluation cost has the same order in arithmetic operations. Appropriate bounded-input and rounded-arithmetic hypotheses add polylogarithmic bits per coefficient.

**Proof.** Let w_i=e^{<k_i,q>}, w_i=p_t(<k_i,q>). From (8), |w_i−w_i|≤η_t w_i for every i, every q, and every realized K, even if K depends on past sketch outputs. Let Z=Σw_i, Z=Σw_i, N=Σw_i v_i, N=Σw_i v_i. Then Z≥(1−η_t)Z>0. By Cauchy-Schwarz,

    ‖N−N‖₂ ≤η_t Z ‖p‖₂‖V‖_F,    ‖N‖₂≤Z‖p‖₂‖V‖_F,
    |Z−Z|≤η_t Z.

Apply Â−A=(N−N)/Z + N(1/Z−1/Z) to obtain (9). All inequalities hold pointwise for all possible input histories, so feedback introduces no additional failure event. ∎

The polynomial-moment idea is **already present in Chen et al.'s Lemma 4.2**; the robust/adaptive consequence is straightforward, not a paper-level novelty claim.

### Space exponent comparison

If B=O(1), d=(log n)^a with fixed a>1, and desired η=n^(−c), c>0 constant, choose

    t = (c+o(1)) log n / loglog n.

Then

    binom(d+t,t) = n^{c(a−1)+o(1)}.                           (10)

For a=2 and c=1/2, deterministic adaptive-robust moments use n^{1/2+o(1)} scalar state, hence o(n) memory (up to polylog factors). In this regime, Chen et al.'s fixed-input/fixed-query upper bound is n^{1/4+o(1)}, up to polylog factors, so the straightforward deterministic route is polynomially more expensive. This is an *upper-bound comparison*, **not** a proved adaptivity lower bound or gap between optimal complexities.

To transfer (9) to complete autoregressive LLM generation, one still needs a bound on the nonlinear propagation of errors through layers into logits (and then through token sampling). The module theorem alone does not prove sequence-level TV control for arbitrary multi-layer Transformers.

## 4. A second robust upper baseline: frozen Gaussian random features with an all-pairs event

There is also a *randomized* streaming summary that survives an adaptively generated KV stream, provided the random-feature approximation is uniform over all key-query pairs in the prescribed balls.

Draw independent ω_j∼N(0,I_d), j=1,...,D, **once** at initialization. Use the identity

    e^{<k,q>} = e^{(‖k‖²+‖q‖²)/2} E_ω[cos(<ω,k−q>)].

Define the 2D-dimensional feature map

    φ_R(x) = e^{‖x‖²/2}/√D · (cos(<ω_j,x>), sin(<ω_j,x>))_{j=1}^D.

Then E_R<φ_R(k),φ_R(q)>=e^{<k,q>}. Maintain only the running feature sums

    S_0=Σ_i φ_R(k_i),     S_v=Σ_i φ_R(k_i) v_i^T.

Return [φ_R(q)^T S_v]/[φ_R(q)^T S_0] whenever the denominator is positive.

### Proposition 3 (standard uniform random-feature argument; proof outline)

For 0<η<1/2 and δ∈(0,1), there exists

    D=Õ(e^{4B} d η^{-2})

with polylogarithmic factors depending on r,d,1/η,1/δ, such that, with probability ≥1−δ over the single random-feature draw,

    sup_{‖k‖,‖q‖≤r} |<φ_R(k),φ_R(q)>/e^{<k,q>}−1|≤η.   (11)

On this **single all-pairs event**, for *every* input stream—including streams chosen adaptively after seeing earlier outputs—and every query, the normalized attention estimate obeys

    ‖Â(q)−A(q)‖₂ ≤ [2η/(1−η)]‖p(q)‖₂‖V‖_F.               (12)

The per-head stored state uses O(D(d+d_v)) scalar numbers including the frozen frequencies, with O(D(d+d_v)) update/evaluation arithmetic up to constants; finite-precision storage and numerical range require additional bookkeeping. With B=O(1), d=polylog(n), η=n^{-c}, and polynomially small δ, the bit-space exponent is n^{2c+o(1)}.

**Proof outline.** For fixed (k,q), every summand before averaging has absolute value ≤e^B. Hoeffding therefore gives absolute kernel error of order e^B√(log(1/δ)/D). Since e^{<k,q>}≥e^{-B}, relative tolerance η follows from D=O(e^{4B}η^{-2}log(1/δ)). To make this simultaneous, cover the two radius-r balls by a finite net. On the high-probability event max_j‖ω_j‖=O(√d+√log(D/δ)), the approximate kernel has a uniform Lipschitz constant of order e^B(r+√d+√log(D/δ)); the exact kernel has a corresponding e^B r bound. A net of mesh proportional to η e^{-B} divided by that Lipschitz constant makes net errors extend to all pairs. The net log-cardinality is O(d log(poly(r,d,D,1/δ)e^{O(B)}/η)). Standard self-consistent choice of D yields the Õ bound. Equation (12) follows from the same numerator/denominator argument as Proposition 2, and crucially the event (11) holds for **every** possible stream, regardless of dependence on the random frequencies. ∎

This is a textbook random-feature + covering argument, **not a new theorem of independent publication value**. It identifies a second robustness envelope alongside deterministic polynomial moments:

    S_rob ≤ Õ(min{(d_v+1) binom(d+t,t), (d+d_v+1)e^{4B}d/η²})

when the Dd frozen random frequencies are counted (and suppressing stated logarithmic bit/precision factors). For a>3 in d=(log n)^a and B=O(1), the second term can be smaller than the straightforward Taylor-moment term. Both may still be polynomially larger than the nonadaptive/discrepancy bound of Chen et al.

Thus the open question is **not** whether *some* adaptive-input robust sublinear-memory sketch exists in bounded-radius regimes: standard uniform-kernel approximations already supply examples. The interesting gap is whether one can retain the much better **discrepancy-based** fixed-query space exponent under full adaptive feedback.

## 5. Scientific decision after this round

**Actually obtained:** an explicit two-step *softmax* feedback witness (not merely an abstract sketch), an accompanying tight information-leakage accounting example, and two constructive adaptive-robust upper baselines (deterministic moments and uniform Gaussian random features) with quantified but nonoptimal memory.

**Not obtained:** an attack against Chen et al.'s actual BaseCompress/Compress implementation; a new space lower bound for general adaptive streaming attention; a near-nonadaptive-space robust randomized algorithm; or a production LLM inference result.

**The next theorem-worthy target:** analyze a *specific* weighted discrepancy coreset under self-generated key/value updates and show one of: (i) a genuine randomized-seed attack in its asymptotic compact-space regime, or (ii) an intrinsic algorithm-specific stability/robustness property that avoids the deterministic moment exponent. A new attack only on arbitrary black-box fixed-query sketches is too easy; a new deterministic moment proof alone is also not enough.

### Primary sources checked

- Chen et al. (2026), *Towards Tight Bounds for Streaming Attention*, https://arxiv.org/html/2606.07205v1 — §3; Lemma 4.2, Lemma 4.3, Theorems 7.1–7.2. 
- Ben-Eliezer et al. (2022), *A Framework for Adversarially Robust Streaming Algorithms*, https://arxiv.org/abs/2003.14265 — standard adversarial streaming model and generic transformations.
- Cohen (2026), *Query-Oblivious Coresets for Softmax Attention*, https://arxiv.org/abs/2609.06327 — all-query coreset construction, not equivalent to optimal adaptive *input-stream* robustness.
