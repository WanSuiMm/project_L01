# Theory status and claim boundaries

This document qualifies the archived theory drafts and the Round 5 discussion.
Correct identities are distinguished from new results and useful certificates.
The real-model pilot adds no uniform theoretical guarantee.

| Claim | Current status |
|---|---|
| A pointwise coreset guarantee implies adaptive accuracy | False in general; Round 2 gives a witness, not a lower bound for Chen's algorithm |
| BaseCompress can be lifted to a whole-query guarantee | Credible covering-net argument under its radius/dimension/freshness assumptions |
| High-temperature space exponent survives the lifting | Conditional on the original parameter regime and polylog overhead |
| Full low-temperature algorithm independently verified | Not established by these notes |
| Same-prefix multilayer KV error recurrence | Conditional stability bound; normalization and heads must be included |
| Output KL and generation TV interfaces | Conditional; no measured sequence-TV endpoint |
| Computable future-safe compression metric on real Transformers | Not established |
| Independent publishable novelty or practical inference advantage | Not established |

## Finite perturbation identities

For a common token prefix, let exact layers be Phi_l and define the actual
compression residual as r_l = Xhat_l - Phi_l(Xhat_(l-1)). With common input,
integration of each layer Jacobian gives an exact decomposition

```math
\Delta z_t=\sum_l G_{t,l}r_{l,t}.
```

The operators retain direction and cross-layer interactions, but depend on the
exact/approximate interpolation paths. This is an identity, not an inexpensive
certificate or an independently identified contribution of each layer.

With p_alpha = softmax((z + alpha Delta z)/tau), define

```math
\mathcal F=2\int_0^1\alpha[\operatorname{diag}(p_\alpha)-p_\alpha p_\alpha^T]d\alpha.
```

Then KL(phat || p) = Delta z^T Fcal Delta z / (2 tau^2), exactly for finite
perturbations. Replacing this path average by Fisher at the original point is
only a local approximation without an additional domination/remainder bound.

If z(u) has Hessian norm at most H along the perturbation segment, and its
first-order change satisfies J u = c 1 exactly, then

```math
D_{KL}(p(u)\Vert p(0))\le H^2\|u\|^4/(16\tau^2).
```

Near-null directions can retain a quadratic leading term. Existence of a large
algebraic KV nullspace does not show that realizable finite eviction/merge errors
lie in it or stay insensitive to future queries.

## A changing metric can hide exponential cost

The sufficient condition D Phi_l^T M_l D Phi_l <= lambda_l^2 M_(l-1), together
with output domination, yields a valid seminorm recurrence. Its inequalities
must hold along the finite interpolation paths and over the prefixes covered
by the guarantee. Semidefinite kernel directions must remain invisible under
the relevant downstream maps.

Even lambda_l <= 1 does not establish a useful depth bound by itself. In the
scalar network Phi_l(x)=2x, choose M_l=4^(L-l). Then every lambda_l can be 1,
while ||r_l||_(M_l)=2^(L-l)|r_l|. Exponential amplification is hidden in the
residual metric. A non-vacuous result must additionally control these residual
norms and the computational cost of obtaining them.

Future-observability Gramians, sum_t J_(t<-s)^T F(p_t) J_(t<-s), are meaningful
local objects on fixed future token paths. They are generally retrospective.
Closed-loop guarantees need a path expectation or uniform bound, and an online
method needs information available before the future is realized. Interacting
errors also require off-diagonal blocks or a conservative composition bound.

## Qualifications to the historical drafts

- The Round 3 bibliography attributes arXiv:2609.06327 incorrectly. Its author
  is Ofek I. Cohen; Liberty, Andoni and Kleiner correspond to arXiv:2605.05602.
- Pre-norm Q/K/V stability must include normalization gains before attention,
  as well as head aggregation and output projection. A post-attention gain
  alone cannot absorb the missing upstream map.
- The sqrt(T) TV bound is a sufficient worst-case interface, not a universal
  unavoidable cost or a necessary practical quality condition.
- A T^(1/4+o(1)) space expression remains conditional on source-theorem regimes,
  value/norm/temperature factors and depth gains. It is not a measured KV
  allocation. KKT allocation optimizes a power-law bound surrogate.
- Partition is an offline operation on a fixed buffer. Conditional success
  can follow from fresh randomness after the buffer is fixed; an implementation
  freshness/composition audit is still needed. This is not a demonstrated
  impossibility of adaptive Partition.
- Reported synthetic numerical checks in the historical drafts are separate
  from the 144-case real-model endpoint and were not re-executed for this release.

## Closest mechanisms

[Behavior-Preserving KV Cache Compression](https://arxiv.org/pdf/2610.06479)
already scores estimated final output KL, accounts for joint eviction
interactions, and compares scores with KL at future positions on reference
continuations. Its compression objective uses available query positions; it
explicitly discusses the difficulty of unseen future states. Output-KL scoring
and a current-score/future-KL diagnostic are therefore not independent novelty.

[The risk of KV cache compression](https://arxiv.org/html/2607.01520v1) uses
query-distribution-weighted response covariance to characterize single-head
attention compressibility. [AATC](https://arxiv.org/html/2608.14191v1) analyzes
attention-output distortion and precision allocation under a quantization-noise
model. [Cohen's query-oblivious coresets](https://arxiv.org/html/2609.06327v2)
provide whole-query constructions; vacuous worst-case upper bounds on measured
geometry are not incompressibility lower bounds.

A prospective contribution would need a cheap, causal and non-vacuous measure
of downstream/future impact for realizable structured errors, with meaningful
comparison to these mechanisms. The first recent-position proxy did not clear
its frozen pilot gate.
