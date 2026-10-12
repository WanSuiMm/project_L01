# Interpretation: one frozen Round 9 half-cache case

**Decision: a full-network first-order response is worth investigating; the
current half-cache certificate remains failed. The dominant observed slack is
upstream propagation and its wide-radius feedback, not solely output rank.**

Read [RESULTS.md](RESULTS.md) for measurements and [verification.json](verification.json)
for the independent offline arithmetic check. This interpretation adds no new
measurements to the frozen run.

## What was selected and run

The mechanical selection rule chose the first original seed (11) and first
n=64 prefix (`histogram-n64-0000`) before the new derivative measurements.
This happens to be the largest-upper-bound clustering case in the previous
15-case half-cache table; it is not an estimate of the average case. Reuse
the exact original width-64 two-layer checkpoint, positive agglomerative m=32
summary, T=4 and all 40 prediction histories. No training or parameter sweep.

The original model was retrieved and hash-checked. The diagnostic ran on local
CPU in 3.48 seconds excluding interpreter/import startup. Each interpolation
decoder generated its own suffix KV for the common discrete history. Endpoint
identity error was <=3.56e-15; the full-unroll JVP/finite-difference discrepancy
was <=8.15e-11 over all hidden/logit coordinates. The selected Round 9 sequence
upper bound replayed to 4.34e-13 and exact KL to 9.1e-17.

## What the measurements distinguish

| Source | Median over 40 histories | Range |
|---|---:|---:|
| Hidden radius / actual hidden error | 166.42 | 114.37–295.07 |
| Final LN bound at actual hidden error / actual normalized error | 1.120 | 1.094–1.220 |
| Centered-head norm estimate / actual centered-logit error | 5.748 | 5.092–15.952 |
| Centered-head row-space fraction of actual normalized error | 0.237 | 0.110–0.299 |
| Exact-logit quadratic KL bound / actual conditional KL | 1.993 | 1.015–11.384 |
| JVP vector remainder / max(endpoint,JVP,1e-10) | 1.850% | 0.991%–6.369% |

Actual final hidden errors are 0.0380–0.1233, while radii are 4.430–36.091.
Thus a substantial gap already exists before the final readout. Directional
readout information helps, but rank <=2 by itself was not the whole mechanism.
The measured row-space fraction is a norm fraction, not a probability or
evidence of unusually strong alignment relative to an isotropic control.

The ratios above are separate diagnostics, not an exact multiplicative
decomposition: the old bound evaluates LayerNorm at a much wider radius.
An exact decomposition of the composed logit-norm gap uses the ratio of the
readout bound at the old radius to the readout bound at the actual radius,
then the final-LN ratio, then the head ratio. The first factor, including
wide-ball LN feedback, has median 264.81 and range 137.80–463.95 in the
offline check. Products are checked per history, not by multiplying medians.

## Oracle substitutions are informative but cannot certify

The existing P-only sequence upper bound is **36.623076**; exact KL(Q||P) is
**3.45822e-6**. Weighting the existing conditional bounds by actual Q history
masses yields **21.605575**, still very loose. This identifies substantial
conditional-bound slack even before any P-to-Q law transfer.

If actual Q hidden-error norms replace the radii while retaining the existing
scalar final-LN/readout machinery, the Q-weighted sum falls to **1.56286e-4**.
Using actual normalized-state norms gives **1.19210e-4**; using actual logit
directions in the quadratic/oscillation/Fisher candidates gives **3.47006e-6**.
These are oracle sensitivity analyses. The Q norms and Q masses are unknown
to the proposed P-only certifier, so none is a new half-cache certificate.

The simple squared norm bound from actual centered logits sums to **9.71330e-6**.
Its remaining gap to exact KL is modest compared with propagation slack.

## What the JVP permits us to conclude

The predeclared <=25% worst-history first-order screen passes: maximum scaled
logit vector remainder is **6.369%**, median **1.850%**; minimum JVP/endpoint
cosine is **0.997980**. Hidden-state scaled remainder is at most **2.652%**.
This supports a first-order description along this specific measure path,
including self-generated suffix KV. The finite response is not dominated by
a large measured endpoint Taylor remainder in this case.

It does not establish a small supremum of the second derivative on [0,1],
a reference-only bound on the remainder, a unique cancellation mechanism, or
generalization across training seeds, prefixes and horizons. T=4 remains a
finite full-tree diagnostic. No claim of a useful certificate, novelty or
systems benefit is added, and the original Round 9 primary gate stays failed.

The next mathematical target would be a P-derived bound on full-network
directional propagation and its nonlinear remainder. Reintroducing per-layer
worst-direction norm products would recreate the measured upstream slack.
This result justifies a bounded attempt at that target, not more model training
or a larger empirical sweep.
