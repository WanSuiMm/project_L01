# Interpretation: directional remainder feasibility fails

**The accurate end-to-end JVP survives, but a norm-only enclosure of the
nonlinear remainder is still too loose. This attempt supplies no useful
half-cache certificate.**

The frozen case is unchanged: original seed 11, first n=64 prefix, m=32
positive agglomerative summary, T=4 and all 40 prediction histories. There
was no model, prefix, budget or horizon search. The bound is built from P and
the fixed summary; prior actual Q measurements are independent validation.
No Q history masses enter the new bound.

## Method and checks

Each node retains its P value, signed path tangent, and a norm upper bound
on its second derivative over theta in [0,1]. Linear maps and residuals keep
signed first-order directions. LayerNorm uses the affine segment plus its
Taylor-remainder ball; GELU uses safe derivative constants. Attention uses
a fixed key gauge, positive denominator enclosure and centered numerator.
Each path generates its own suffix KV. The
[method note](source/ROUND9_DIRECTIONAL_REMAINDER_V1.md) derives the inequalities.

Four focused software tests passed. In the scientific run, P logit value
parity error was <=3.56e-15 and prior JVP parity error <=4.52e-16, including
hidden states. All 40 recorded endpoint Taylor inequalities passed. The CPU
scientific calculation took 1.36 seconds, excluding interpreter/import startup.
The [offline audit](verification.json) reconstructs hashes, Taylor arithmetic,
intersections, all 81 path risks, entropy-transfer grid, statistics and verdict.
It does not replay weights or verify a derivative supremum. Ordinary float64
arithmetic and spectral norms are not outward-rounded proof values.

## Result

| Quantity | Value |
|---|---:|
| Original P-only sequence KL upper | 36.623076 |
| New upper intersected with original conditional bounds | 30.905579 |
| Frozen feasibility threshold | 0.1; FAIL |
| Actual sequence KL, validation only | 3.45822e-6 |
| Improved conditional bounds | 27/40 |
| Finite second-derivative bounds | 40/40 |
| Median / maximum Taylor-remainder norm upper | 1.68686 / 1.17287e9 |

Without the old-bound intersection, the raw directional upper is 3.43908e17.
Median remainder upper / actual endpoint remainder is about 30,625; the largest
ratio is 8.11e12. All bounds are finite here. This is enclosure looseness,
not an interrupted computation or infinity silently clipped into a pass.

## Where growth occurs

History (0,2) has the largest centered-logit second-derivative upper:

| Stage at its last prediction | Second-derivative norm upper |
|---|---:|
| First block attention | 0.006927 |
| First block after MLP residual | 0.066595 |
| Second block attention | 2.042696 |
| Second block after MLP residual | 175.894922 |
| Final LayerNorm | 2.55604e9 |
| Centered logits | 2.34575e9 |

Its measured endpoint logit remainder is only 1.44701e-4. The broad MLP-residual
bound feeds the final LayerNorm enclosure, where the largest jump occurs.
This is consistent with the enclosure approaching its epsilon-limited worst
case, but the internal minimum-variance bound was not retained. It does not
show that the true path approaches zero variance.

Some extreme histories contain continuation symbol 2 and have small P mass:
P(0,2)=3.52730e-5 and P(2,2)=9.25931e-9. Full three-token support must remain.
However, the empty-history conditional upper is already **3.32726**, above
0.1 before any sampled token or P-to-Q history-law transfer. Rare histories
are therefore not the entire failure; removing symbols changes the task.

## Consequence

This closes the inexpensive attempt to preserve first-order direction but
replace every second-order uncertainty by a scalar norm. The previous maximum
measured JVP remainder of 6.37% remains correct; accurate linearization does
not automatically yield a tight computable supremum enclosure.

Further constant tuning or a larger model sweep is not supported. A new
attempt would need more structure in the nonlinear remainder or a narrower
property that avoids these worst-direction products. That is a materially
different method, not a successful reinterpretation of this run. No claim is
made that all directional methods are impossible, earlier theory is false,
or L01 novelty is established. Half-cache high-fidelity certification remains
unresolved.
