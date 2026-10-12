# Round 11 streaming covariance results

Runner completion: **COMPLETE**; metric-grid coverage: **COMPLETE**. Primary rows: 960/960; Fisher rows: 60/60

Endpoint A status: **ROUND10_ORDER_TRANSFER_WITH_LOCAL_COMPARABILITY_SCREEN** at T=128. This is a descriptive transfer screen over trained model seeds.

Endpoint B status: **FINITE_FISHER_EXPLANATION_SUPPORTED** at amplitude 0.25. It compares a finite reverse sequence KL with full and diagonal local Fisher predictions.

The sequence estimand averages fixed-schedule KLs. It is not the KL of a token law marginalized over hidden compression schedules. Model seed is the independent unit; prompts, schedule draws, paths, and Fisher probe histories are averaged within seed.

## Endpoint A at the primary horizon

| Model seed | Coherent KL | IID KL | Alternating KL | Order pass | Local error max/min | Local screen |
|---:|---:|---:|---:|:---:|---:|:---:|
| 11 | 2.80259 | 0.211126 | 0.11155 | yes | 1.00554 | yes |
| 23 | 2.81503 | 0.180205 | 0.0994903 | yes | 1.01013 | yes |
| 37 | 2.67452 | 0.160183 | 0.0934827 | yes | 1.00653 | yes |
| 51 | 2.66738 | 0.175002 | 0.0915571 | yes | 1.01176 | yes |
| 71 | 2.6914 | 0.148536 | 0.106074 | yes | 1.01071 | yes |

The paired coherent-minus-alternating mean difference is 2.62975; the descriptive 95% training-seed bootstrap interval is [2.57981, 2.68374] (10,000 fixed resamples).

Figure: [mean-seed horizon curves](analysis/sequence_kl_horizon.png), [local distortion versus KL at T=128](analysis/local_distortion_vs_kl.png).

## Endpoint B Fisher diagnostic

At amplitude 0.25, the screen requires the full-Fisher mean relative error to be below the diagonal-only error and at most the configured threshold for both coherent and alternating schedules, in the required number of model seeds.

| Model seed | Policy | Full relative error | Diagonal relative error | Finite reverse KL | Full Fisher prediction | Diagonal Fisher prediction | Pass |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 11 | coherent | 0.000485592 | 0.79223 | 0.008102 | 0.00809817 | 0.00165939 | yes |
| 11 | alternating | 3.58797e-05 | 0.665261 | 0.000995272 | 0.000995236 | 0.00165939 | yes |
| 23 | coherent | 0.000597629 | 0.783358 | 0.00856972 | 0.00856462 | 0.00185429 | yes |
| 23 | alternating | 0.000243366 | 0.372715 | 0.00136578 | 0.00136546 | 0.00185429 | yes |
| 37 | coherent | 0.000531019 | 0.806756 | 0.00976886 | 0.00976369 | 0.00188617 | yes |
| 37 | alternating | 0.000122686 | 1.01972 | 0.00108653 | 0.0010864 | 0.00188617 | yes |
| 51 | coherent | 0.000316077 | 0.802801 | 0.00814852 | 0.00814596 | 0.00160072 | yes |
| 51 | alternating | 6.26212e-05 | 1.12522 | 0.000759426 | 0.000759377 | 0.00160072 | yes |
| 71 | coherent | 0.000474597 | 0.79772 | 0.00863516 | 0.00863101 | 0.00174654 | yes |
| 71 | alternating | 0.000138231 | 0.741929 | 0.00100245 | 0.00100231 | 0.00174654 | yes |

Figure: [Fisher predictions versus finite reverse KL](analysis/fisher_predictions.png). Predictions are multiplied by amplitude squared; no fit or proxy is applied.

## Completion and checks

Configured grid: 960 primary rows and 60 Fisher rows. Observed unique rows: 960 and 60. Runner progress: actual_risk; failure type: none recorded.

Software-check references: [streaming core](source/round11_streaming_core.py), [targeted core tests](source/test_round11_streaming_core.py). The analyzer did not rerun them. Runner-recorded checks: none recorded in provenance.

Source hashes come from runner provenance; checkpoint and prefix hashes come from the separate run manifests. The report retains only file or prompt labels and SHA-256 digests, not raw prefixes or machine paths. See summary.json for coverage diagnostics, all five model-seed values, local feature/denominator summaries, JVP remainder, parity gaps, and available signatures.

These screens do not establish a trained-Transformer theorem, a certificate, a new compressor, an asymptotic rate, or an inference speedup. An ordering without the local-distortion comparability screen remains confounded.
