# Original Round 6 local-bound evaluation

Status: COMPLETE; FLOAT64_ANALYTIC_BOUND_EVALUATION_NOT_INTERVAL_CERTIFIED.

Completed 330/330 conditions. No training performed.

| n | m | Method | Mean upper KL(Q→P) | Mean actual KL(Q→P) | Prefixes <=0.1 |
|---:|---:|---|---:|---:|---:|
| 16 | 4 | agglomerative | 32.6746 | 3.08703e-05 | 0/15 |
| 16 | 4 | gs_thin | 37.6557 | 0.036675 | 0/15 |
| 16 | 4 | uniform_subset | 37.6557 | 0.068581 | 0/15 |
| 16 | 8 | agglomerative | 25.1019 | 1.63507e-05 | 0/15 |
| 16 | 8 | gs_thin | 37.6557 | 0.0113534 | 0/15 |
| 16 | 8 | uniform_subset | 37.6557 | 0.013781 | 0/15 |
| 16 | 12 | agglomerative | 11.4152 | 4.08342e-06 | 0/15 |
| 16 | 12 | uniform_subset | 37.6557 | 0.0236213 | 0/15 |
| 16 | 15 | agglomerative | 0.265307 | 3.37138e-07 | 5/15 |
| 16 | 15 | uniform_subset | 37.6557 | 0.00474644 | 0/15 |
| 16 | 16 | full | 1.27355e-28 | 0 | 15/15 |
| 64 | 16 | agglomerative | 11.8846 | 7.04315e-06 | 0/15 |
| 64 | 16 | gs_thin | 37.4676 | 0.00750011 | 0/15 |
| 64 | 16 | uniform_subset | 37.4676 | 0.021981 | 0/15 |
| 64 | 32 | agglomerative | 8.4763 | 5.43662e-06 | 0/15 |
| 64 | 32 | gs_thin | 37.4676 | 0.00338083 | 0/15 |
| 64 | 32 | uniform_subset | 37.4676 | 0.0221668 | 0/15 |
| 64 | 48 | agglomerative | 4.13743 | 3.22723e-06 | 0/15 |
| 64 | 48 | uniform_subset | 37.4676 | 0.00233797 | 0/15 |
| 64 | 63 | agglomerative | 0.00659081 | 1.85212e-08 | 15/15 |
| 64 | 63 | uniform_subset | 21.5685 | 0.000196651 | 0/15 |
| 64 | 64 | full | 0 | 0 | 15/15 |

Read summary.json for all five seed results and the predeclared primary gates.
Bounds are float64 evaluations, not strict interval certificates. Record reductions are not measured memory savings.
Per-history files are secondary evidence; do not open them before the aggregate.
