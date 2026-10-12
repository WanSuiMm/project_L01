# Directional second-derivative norm enclosure: frozen one-case result

Execution COMPLETE; numerical status FLOAT64_ANALYTIC_DERIVATIVE_BOUND_EVALUATION_NOT_INTERVAL_CERTIFIED.

Seed 11, histogram-n64-0000, n=64,m=32,T=4, positive agglomerative summary; 40 histories, no training.
The bound is constructed from P/reference quantities and the fixed summary. Actual Q is validation only.

**Frozen sequence KL<=0.1 feasibility target: FAIL.**

Old Round 9 sequence upper: 36.623076.
New directional bound intersected with old conditional bounds: 30.9055793.
Exact sequence KL (independent prior diagnostic): 3.45821894e-06.
Unbounded second-derivative evaluations: 0/40.
Histories improved over the old conditional bound: 27/40.

## Second-order enclosures

| Metric | Minimum finite | Median finite | Maximum finite |
|---|---:|---:|---:|
| second_derivative_norm_upper | 0.104811334 | 3.37371497 | 2.3457483e+09 |
| Taylor_remainder_upper | 0.0524056671 | 1.68685749 | 1.17287415e+09 |
| remainder_bound_over_actual | 4057.03435 | 30625.0037 | 8.10547982e+12 |
| raw_directional_conditional_KL_upper | 0.000706440236 | 0.714971092 | 3.43908442e+17 |

Finite-only statistics are not evidence of coverage when some histories have infinite bounds.
histories.jsonl retains all histories and stage-growth diagnostics; read summary.json before opening raw rows.

## Checks and scope

Value/tangent parity with the original P implementation and prior full-network JVP is checked at all histories.
Recorded endpoint Taylor inequalities are independently validated against the previous diagnostic.
A failed enclosure does not invalidate the accurate JVP or prove every directional enclosure impossible.
A numerical pass would still require outward-rounded arithmetic for a strict computer-assisted certificate.
No new model, prefix, horizon grid, training, throughput measurement or novelty audit was performed.
