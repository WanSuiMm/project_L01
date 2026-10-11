# L01 Round 9 — Reference-only path-local certificates for one-time prefix KV compression

**2026-10-11. Status: conditional mathematical theorem + actually executed finite-horizon numerical audit.** The ideal-real-arithmetic inequalities are rigorous under the hypotheses below. The resulting `float64` numerical values are **NOT interval-arithmetic / machine-checked certificates**. The user's original Round 6 two-layer checkpoints are not in the public GitHub repository or available files: this release cannot assert a numerical certificate for those weights. The numerical smoke uses a *separately trained* CPU 2-layer toy (16 hidden units, 400 training steps) and is labeled accordingly.

## 1. Research decision and change from Round 8

Round 8's tube bound remains conditional on verifying score spread, LayerNorm variance, centered prefix response, and downstream gains for **all** relevant histories and interpolation states. Naively sampling the tube is not certification. This round develops a **reference-only**, finite-vocabulary, finite-horizon, deterministic a posteriori procedure. It needs only (i) a trained decoder checkpoint; (ii) a single dense prefix; (iii) one fixed positive mass-preserving prefix summary; (iv) the complete model's reference trajectories for all possible token histories, which are finitely enumerable for a short horizon. It does not use the compressed model's trajectory to calculate the risk bound. A compressed-model replay is only an independent numerical check.

**Important support correction:** the Round 6 histogram decoder has a 3-token full-softmax vocabulary `{0,1,CONT}`. It can generate CONT with positive probability. A rigorous *unconditional* full-sequence guarantee requires covering all `3^T` histories, not only binary `2^T`. The executable enumerates all `1+3+...+3^(T-1)` prediction prefixes.

## 2. Exact path-local LayerNorm gain

For LayerNorm of width d, let `P = I - 11^T/d`, learnable gamma, epsilon>0. The Jacobian operator norm at x satisfies

\[
 \|D\mathrm{LN}(x)\|_{2\to2}\le \frac{\|\gamma\|_\infty}{\sqrt{\|Px\|^2/d+\epsilon}}.
\]

For any `x'` with `||x'-x||<=R`, all points on the line segment stay in the ball and

\[
\boxed{\|\mathrm{LN}(x')-\mathrm{LN}(x)\|_2\le
 L_{\mathrm{LN}}(x,R)R,
\quad L_{\mathrm{LN}}(x,R)=\frac{\|\gamma\|_\infty}
{\sqrt{[\max(0,\|Px\|/\sqrt d-R/\sqrt d)]^2+\epsilon}}.}
\tag{9.1}
\]

The gain is **computed from one known reference state and a proven discrepancy radius**; using the observed variance of a sampled continuation without the radius subtraction would not be valid.

**Necessary obstruction for scalar balls:** if `R>=||Px||`, the ball contains a constant-vector state (`Px'=0`). The local derivative upper bound necessarily returns to `||gamma||/sqrt(epsilon)` in the absence of finer directional information. Making the radius estimate a little less loose will not necessarily resolve this boundary.

## 3. A reference-only finite-attention perturbation lemma

At a fixed layer/time, construct an intermediate attention calculation with the **compressed fixed prefix** but the **full model's reference suffix keys and values and reference query**. This is known using only `P` and the summary. Let its atoms be `(k_i,v_i,w_i)`, probabilities `p_i`, and output `a=sum_i p_i v_i`. These atoms include the summary atoms with positive weights, plus one weight-one suffix atom per observed input position.

The compressed model has the **same summary atoms** and unknown suffix/query perturbations satisfying

\[
\|\delta q\|\le \zeta,\quad
\|\delta k_i\|\le\kappa_i,\quad
\|\delta v_i\|\le\nu_i,
\]

where `kappa_i=nu_i=0` on fixed prefix-summary atoms. An arbitrary common logit shift is immaterial. Choose any fixed key center `c` (we use the known reference probability-weighted center), set

\[
U_i=\zeta\|k_i-c\|+(\|q\|+\zeta)\kappa_i.
\]

Then **after subtracting the same harmless score shift**, the unknown score change for atom i is in `[-U_i,+U_i]`. Define the known scalars

\[
Z_- =\sum_i p_i e^{-U_i},\qquad Z_+=\sum_i p_i e^{U_i},
\quad
\underline p_i=\frac{p_i e^{-U_i}}{Z_+},\quad
\overline p_i=\frac{p_i e^{U_i}}{Z_-}.
\]

Consequently all perturbed probabilities satisfy `underline p_i <= p_i' <= overline p_i`. Center the value term at `a` and use positive weights:

\[
\boxed{
\|a'-a\|\le
\sum_i \overline p_i\nu_i+
\sum_i\max\{p_i-\underline p_i,\overline p_i-p_i\}\|v_i-a\|.
}
\tag{9.2}
\]

This is a **strict endpoint bound** in exact real arithmetic, not a small-perturbation Taylor expansion. Since logit scores are defined up to a common shift, `k_i-c` matters; an enormous shared key translation incurs no artificial instability. The proof is `a'-a = sum_i p_i' (v_i'-v_i) + sum_i (p_i'-p_i)(v_i-a)` and the probability intervals above. The implementation also uses a safe bounded-value fallback to avoid overflow.

The prefix forcing at *reference* query and suffix is directly computable:

\[
 b_{\ell,t}=\big\|\operatorname{Attn}(\widehat M_\ell^0, K^P_{\mathrm{suffix}}, V^P_{\mathrm{suffix}}, q^P_t)
 -\operatorname{Attn}(M_\ell^0, K^P_{\mathrm{suffix}},V^P_{\mathrm{suffix}},q_t^P)\big\|.
\tag{9.3}
\]

It is not fitted to final KL, and uses no future generated-token trace: its value is calculated separately for **each enumerated hypothetical history** from full-model conditional computation.

## 4. Constructive reference-only two-layer recursion

For each of the finitely many prefix histories `u`, compute the reference model's full causal trajectory along `u` and the fixed cache summary. This gives the exact reference states `x^P_{l,s}`, per-layer queries and suffix key/values, and post-attention residual intermediates `y^P_{l,s}`. Define recursively

\[
 E_{0,s}=0,
\]

and for layer l at time t, first calculate the query, suffix-key and suffix-value uncertainty radii using (9.1) and the *previous-layer* proven errors `E_{l-1,s}`:

\[
\zeta_{l,t}=\|W_{Q,l}\|L_{LN,l}(x^P_{l-1,t},E_{l-1,t})E_{l-1,t},
\]

\[
\kappa_{l,s}=\|W_{K,l}\|L_{LN,l}(x^P_{l-1,s},E_{l-1,s})E_{l-1,s},
\quad\nu_{l,s}=\|W_{V,l}\|L_{LN,l}(x^P_{l-1,s},E_{l-1,s})E_{l-1,s}.
\]

Here `1/sqrt(d_key)` is distributed symmetrically between Q and K, as in the implementation. Apply (9.2) to get `B_{l,t}` for actual-summary vs reference-summary suffix/query changes; add the exactly computed forcing (9.3). For the pre-LN residual block

\[
 y_l=x_{l-1}+W_{O,l}a_l,
\quad x_l=y_l+W_{2,l}\mathrm{GELU}(W_{1,l}\mathrm{LN}_{ff,l}(y_l)+b_{1,l})+b_{2,l},
\]

let `D_{l,t}=E_{l-1,t}+||W_O|| [b_{l,t}+B_{l,t}]`. The resulting **reference-only induction** is

\[
\boxed{E_{l,t}=D_{l,t}\left[1+
\|W_{2,l}\|\,L_{\mathrm{GELU}}\|W_{1,l}\|
L_{LN,ff,l}(y^P_{l,t},D_{l,t})\right],}
\tag{9.4}
\]

where `L_GELU<=1.13` for exact GELU. Valid range caps (bounded LN output, bounded value magnitude) can be intersected with these inequalities. This is not circular: calculate E in ascending layer order, with all previous-layer time positions already known from induction. No unknown `Q` state is inserted into a supposed local constant. For one-time prefix replacement the old prefill layer caches are fixed; suffix KV are each model's own.

At the readout, define `R_t=E_{L,t}L_{LN,final}(x^P_{L,t},E_{L,t})` (again potentially clipped by a **true** bounded-LN output-range cap), and set

\[
\Omega_t=R_t\max_{a,b}\|W_{LM,a}-W_{LM,b}\|.
\]

This explicitly respects the output-softmax common-shift invariance. Given the exact reference next-token distribution `p_t`, set

\[
 F_t=\operatorname{diag}(p_t)-p_t p_t^T,
\quad G_t=\lambda_{\max}(W_{LM}^T F_t W_{LM}).
\]

By the Softmax Bregman integral and `p_{t,alpha}(a)/p_t(a)<=exp(Omega_t)` one obtains **both conditional KL directions** bounded by

\[
\boxed{r_t^{\mathrm{both}}(u)=\min\left\{
\frac{\Omega_t^2}{8},\
\frac12 e^{\Omega_t}G_tR_t^2
\right\}.}
\tag{9.5}
\]

Additionally, **only in the forward `Q||P` direction**, one may intersect with the known bound `-log min_a p_t(a)`, obtaining `r_t^+(u)`. Using that reference cap for `P||Q` would be unsound. These finite-history bounds are *reference-only*; `Q` evaluations are not needed to construct them.

## 5. Full-sequence certificate from the entire reference probability tree

Let `P` denote the full prefill model and `Q` the fixed compressed prefill model, each subsequently appending its own KV. Enumerate **all** histories in the actual full-softmax alphabet \(\mathcal V\), and let `r_t^+(u)` / `r_t^-(u)` be the reference-only conditional bounds from §4. For each complete length-T output string define

\[
R^\pm(y_{1:T})=\sum_{t=1}^T r_t^\pm(y_{<t}).
\]

From the KL chain rule (for both directions separately) and the pointwise conditional bounds:

\[
D_{KL}(P\Vert Q)\le \mathbb E_P R^-.
\tag{9.6}
\]

For `D(Q||P)`, the paths are sampled under the **unknown perturbed distribution** Q; pointwise bounds cannot simply be averaged under P. But for any `lambda>1`, the entropy variational inequality yields

\[
\lambda\mathbb E_Q R^+\le D_{KL}(Q\Vert P)+\log\mathbb E_P e^{\lambda R^+}.
\]

Since `D_KL(Q||P)<=E_Q R^+`, absorption produces the *reference-only full-law certificate*

\[
\boxed{
D_{KL}(Q\Vert P)\le
\min\left\{
\max_y R^+(y),\
\inf_{\lambda>1}\frac{\log\mathbb E_P e^{\lambda R^+(Y)}}{\lambda-1}
\right\}.}
\tag{9.7}
\]

**There is no small-risk `v<1/4` assumption in this finite-tree result.** Its cost is exponential in the horizon, and without tight per-history local radii it may still be vacuous. The reference expectation is calculated with exact P conditional probabilities, on **all** full-support histories. The probability weights and conditional bounds are assembled by a finite recursion; sampling/one-trajectory teacher forcing would not establish (9.7).

This is a finite-horizon, *a posteriori checkpoint/summary-specific mathematical guarantee*, not a cheap deployment-time prior certificate and not an efficient arbitrary-length theorem. The layer-local softmax and LayerNorm verification components overlap substantially with prior NN robustness/abstract-interpretation research; it should not be called an original main theorem without stronger novelty audit.

## 6. Code and actually executed small tests

The provided `round9_certify.py` faithfully implements the Round 6 `TinyCausalDecoder` architecture (pre-LN, genuine softmax, learned absolute positions, residual attention, exact GELU, residual MLP, final LN, 3-token full softmax) in float64. It can load an actual saved Round 6 checkpoint using `--checkpoint` and `--project-root` on the machine holding that private file. The original `round6_transformer.py` is imported to assert matching model/schema before loading its weights.

The originally published Round 6 checkpoints (e.g. `histogram_d64_l2_seed11/checkpoint.pt`) are **excluded from GitHub and unavailable to this run**. Therefore the numerical tests below are **not on the user's published Round 6 trained models**. The saved smoke fixture is a separately trained, 2-layer, width-16 Histogram causal transformer: 400 AdamW steps, n=16, suffix=8; an additional 512-episode validation has `model NLL 0.61293`, Bayes NLL `0.60946`, gap `0.00347` in the shortened task. It is NOT part of Round 6's formal 10 qualified checkpoints and does not replace their frozen results.

**One fixed CPU-trained fixture, one fixed n=16 prefix, T=4; full vocabulary (3 symbols):**

| Fixed prefix summary (all positive, mass 16) | Retained atoms | Reference-only candidate upper bound on Q||P | Exact Q||P (independent replay) | Reading |
|---|---:|---:|---:|---|
| Closest pair merge | **15** | **0.0196574** | **3.4810e-7** | nonvacuous (<1), but ~56,470× loose |
| Agglomerative cardinality-weighted KV clusters | **13** | **0.0714920** | **3.9e-6** | nonvacuous (<1), 18.75% old-prefix record reduction |
| Same agglomerative construction | **11** | **0.909479** | **4.4e-6** | barely nonvacuous (<1), 31.25% reduction |
| Same agglomerative construction | **8** | **15.8733** | **4.7e-5** | vacuous for a useful KL<1 goal |
| Eight disjoint greedy KV pairs | **8** | **26.2443** | **0.00221275** | almost identical to reference-only trivial cap 26.2485 |

The compressor selects/merges records **using only the prefix KV**; no future token outcomes are used. These are not two different random seeds of GS-Thin: `agglomerative` produces synthetic weighted representatives and is **outside** the Round 7 equal-weight original-subset minimax class. This study is an existential local-certificate audit, **not evidence of GS-Thin's theory in trained models**. The finite budget table is **not monotonic**: the summary itself changes when m changes, so neither the observed KL nor the bound is required to be monotone across these specific greedy summaries.

The theorem is sound in exact arithmetic, but its numerical bound remains much looser than measured KL. `m=11` is important only because it proves a *numerically nontrivial* working configuration at an actual 31.25% prefix-record reduction on an independent CPU-trained model; it is not yet an efficiently computed interval certificate or a reusable large-budget compression theorem.

A second diagnosis is particularly informative. At `m=15`, maximum local ball radius divided by the full-reference centered-state norm was **0.0419**, and no analysed LayerNorm uncertainty ball intersected the zero-variance set. At `m=8` with **disjoint greedy pairs**, the maximum ratio was **2.033**, and **39** layer/token/history tube checks intersected that set. Hence, scalar-ball verification can become vacuous by failing to rule out the LayerNorm singular region. However, `m=8` **agglomerative** still fails the KL<1 goal with maximal ratio **0.365** and **zero such intersections**. The latter demonstrates that even when the normalization denominator is controlled, scalar uncertainty propagation and downstream gains can still be uninformatively large. These are two distinct sources of slack; neither is a theorem violation.

Each `T=4` audit covers `1+3+9+27=40` reference prediction histories and all `3^4=81` possible full-softmax sequences. No Q conditional probabilities are used to form the bound; the exact Q replay is optional and checks the inequality. Four implemented unit tests passed, including 200 randomized endpoint attention perturbation checks and 300 LayerNorm chord checks, plus 18 additional small end-to-end random-seed checks. These are not machine-checking or a substitute for a sound interval implementation.

**Critical numerical qualification:** this code uses float64 spectral norms, softmax, GELU, and probabilities; no correctly outward-rounded interval arithmetic is implemented. Consequently these reported finite decimal upper bounds are *numerically evaluated analytic bounds*, not certified decimal enclosures. For a claim of an actually strict computer-assisted numerical certificate, each matrix norm, reference activation, exp/log/softmax and accumulated bound must be enveloped using directed rounding / interval arithmetic, with verified checkpoint tensor interpretation and FP-model conventions.

## 7. Next logical research step and decision gate

The next step should **not** be another broad training sweep. On the already trained Round 6 two-layer checkpoint, use one frozen prefix (n=16) and one single positive summary (m=15 for a first existential qualification), T=3 or 4, enumerate all 3-ary output histories, and run this verifier. If the reference-only estimate is <1, then repeat at m=8 with equal-weight thinning (GS-Thin or uniform) and compare absolute certs. The code is ready for the private checkpoint but it is not accessible here.

The mathematical bottleneck found by this independent CPU fixture is specific: once `E_{l-1,t} >= ||P x^P_{l-1,t}||`, scalar balls cannot rule out LN zero variance, and the local bound reverts to the singular global worst case. A defensible next improvement would be **directional state enclosures** (e.g. affine forms / zonotopes) or tightly verified secant-Jacobian action, rather than tuning scalar `C_*`. Such verification methods already have substantial prior art, so resulting methods need a genuine, source-specific distinction to count as NAACL theory novelty.

If the original checkpoint's m=15/T=4 certificate cannot be made nonvacuous with reference-only state information, this line still has no evidence of an effective learned-model verifier. If m=15 passes but m=8 fails, the problem is **tightness under meaningful compression**, not just the possibility of any nonempty local certificate.

## 8. Source/novelty references (not new mathematical ingredients)

- L01 Round 6/7/8 user-supplied notes and repository state `7558461` (exact prefix replacement, trained Histogram evidence, global-vacuity audit).
- Shi et al., *Robustness Verification for Transformers*, 2020: https://arxiv.org/abs/2002.06622 . Established Transformer certification and attention cross-position bounds.
- Rezazadeh & Davoodi, *Vertex-Softmax: Tight Transformer Verification via Exact Softmax Optimization*, 2026: https://arxiv.org/abs/2605.10974 . Examines exact score-box optimality; our elementary score interval bound is not claimed as novel or sharp.
- Gelberg et al., *Training Transformers for KV Cache Compressibility*, 2026: https://arxiv.org/abs/2605.05971 . Prefix-cache replacement and representation dependence predate L01.

**Scientific verdict:** The finite-history reference-only method has a sound *mathematical* inequality proof and a separately trained two-layer CPU model exhibits nontrivial numerical upper bounds at m=15/13/11 out of n=16 (up to 31.25% record reduction). The requested **original existing** Round 6 two-layer checkpoint remains **unverified because the weights are unavailable here**. At m=8 the scalar-radius verifier fails even on this small trained fixture. The computation is float64, **not a machine-certified interval enclosure**, and the observed bound is far from tight relative to measured KL. The next theoretical challenge is directional or affine-form reachable-state verification. This is a research qualification, not a publishable independent theorem by itself.
