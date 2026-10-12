# Round 14 frozen associative-recall transfer analysis

Scientific verdict: **TASK_UNQUALIFIED**.
Metric coverage: **True** (160/160 rows); P-quality coverage: **True** (10/10 rows).

The independent unit is the model seed (5 seeds). Each seed's values average the two evaluation prompts and two phase schedules equally. The 16 sampled paths per phase are within-seed Monte Carlo records, not independent seeds; there is no antipodal path count.

Readiness gates retain every configured seed. Training quality uses the model-level `quality_pass` and available final-validation top-1 metrics; P-generation quality pools answer correctness over all answer positions and averages query-valid rates over prompts (or pools explicit query counts when present).

Training qualified: **4/5**; P-generation gates passed: **4/5**.

| Seed | Train answer top-1 | Train first-answer top-1 | Train gate | P query-valid | P answer accuracy | P gate |
|--:|--:|--:|:--:|--:|--:|:--:|
| 11 | 1 | 1 | PASS | 0.999519 | 0.999756 | PASS |
| 23 | 1 | 1 | PASS | 0.999279 | 0.999023 | PASS |
| 37 | 0.42485 | 0.435547 | FAIL | 0.978846 | 0.362061 | FAIL |
| 51 | 1 | 1 | PASS | 1 | 1 | PASS |
| 71 | 1 | 1 | PASS | 1 | 1 | PASS |

Endpoint horizon: **129**. Fisher screen: per-seed mean relative error ≤ 0.25 in at least 4/5 seeds, and full Fisher closer to actual KL than diagonal Fisher in at least the same number.

| Policy | Mean actual KL | Mean full Fisher | Mean diagonal Fisher | Full error passes | Full better than diagonal |
|:--|--:|--:|--:|--:|--:|
| balanced_block | 0.000668265 | 0.000567869 | 0.000302254 | 1/5 (FAIL) | 4/5 (PASS) |
| alternating | 0.000274418 | 0.000246408 | 0.000301715 | 3/5 (FAIL) | 2/5 (FAIL) |

Per-seed balanced-block / alternating ratios at the endpoint:

| Seed | Actual KL | Full Fisher | Diagonal Fisher | Histogram-bias area |
|--:|--:|--:|--:|--:|
| 11 | 1.45093 | 1.06174 | 0.997784 | 4.12922 |
| 23 | 1.59774 | 1.23103 | 0.999541 | 4.50236 |
| 37 | 2.72281 | 2.61509 | 1.00216 | 4.32034 |
| 51 | 1.83508 | 1.67825 | 1.00095 | 4.47801 |
| 71 | 1.8681 | 1.04887 | 1.00206 | 4.42878 |

Full-Fisher error dispersion across model × prompt × phase-schedule cells, separately for every policy and horizon. These cells are nested within seeds and are not independent replicates:

| Policy | Horizon | Cells | Abs. error MAE | Median | Maximum | Rel. error MAE | Median | Maximum |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| balanced_block | 17 | 20 | 5.04139e-06 | 1.98977e-06 | 2.19632e-05 | 0.360598 | 0.385742 | 0.799103 |
| balanced_block | 33 | 20 | 1.98838e-05 | 6.75912e-06 | 9.3214e-05 | 0.418753 | 0.425646 | 0.848277 |
| balanced_block | 65 | 20 | 7.68025e-05 | 3.66991e-05 | 0.0002881 | 0.437852 | 0.405179 | 0.869469 |
| balanced_block | 129 | 20 | 0.000119732 | 6.15704e-05 | 0.000430983 | 0.337495 | 0.25991 | 0.79614 |
| alternating | 17 | 20 | 2.69249e-06 | 1.55078e-06 | 9.87515e-06 | 0.221883 | 0.13995 | 0.648794 |
| alternating | 33 | 20 | 7.36901e-06 | 3.79024e-06 | 2.05717e-05 | 0.233603 | 0.16021 | 0.672757 |
| alternating | 65 | 20 | 1.69671e-05 | 1.01001e-05 | 6.20118e-05 | 0.237626 | 0.133836 | 0.628355 |
| alternating | 129 | 20 | 2.87484e-05 | 2.0318e-05 | 9.34604e-05 | 0.212549 | 0.190493 | 0.543799 |

Dispersion after averaging prompts and schedules within each seed:

| Policy | Horizon | Abs. error MAE | Median | Maximum | Rel. error MAE | Median | Maximum |
|:--|--:|--:|--:|--:|--:|--:|--:|
| balanced_block | 17 | 3.89194e-06 | 4.06398e-06 | 6.93332e-06 | 0.277738 | 0.310633 | 0.504937 |
| balanced_block | 33 | 1.6277e-05 | 1.35398e-05 | 3.65213e-05 | 0.348862 | 0.38048 | 0.660047 |
| balanced_block | 65 | 5.8449e-05 | 6.02904e-05 | 0.000121724 | 0.419731 | 0.493776 | 0.713493 |
| balanced_block | 129 | 0.000100397 | 9.52592e-05 | 0.000186742 | 0.372717 | 0.422912 | 0.643638 |
| alternating | 17 | 2.55133e-06 | 1.80917e-06 | 5.4243e-06 | 0.224068 | 0.251268 | 0.454341 |
| alternating | 33 | 7.04505e-06 | 4.16401e-06 | 1.48852e-05 | 0.236892 | 0.23014 | 0.536842 |
| alternating | 65 | 1.62828e-05 | 9.74863e-06 | 3.52338e-05 | 0.237831 | 0.224703 | 0.52323 |
| alternating | 129 | 2.80101e-05 | 1.98778e-05 | 5.67349e-05 | 0.212853 | 0.211372 | 0.44299 |

Local-attention error comparability is descriptive, computed as max/min of the two policy means within each seed at the endpoint:

| Seed | Block mean | Alternating mean | Max/min | ≤1.25 |
|--:|--:|--:|--:|:--:|
| 11 | 0.232918 | 0.146076 | 1.5945 | FAIL |
| 23 | 0.0374486 | 0.0313934 | 1.19288 | PASS |
| 37 | 0.0735212 | 0.0544211 | 1.35097 | FAIL |
| 51 | 0.412383 | 0.29173 | 1.41358 | FAIL |
| 71 | 0.558928 | 0.27799 | 2.0106 | FAIL |
1/5 assessed seeds meet the 1.25 ratio; this descriptive check does not enter the primary verdict.

Full and diagonal Fisher predictions are evaluated on sampled Q1 histories; they are not P-law sequence Fisher estimates or held-out predictions.

The scalar histogram-area baseline fits one no-intercept coefficient on the other four seed means, pooling both policies, then predicts both held-out policy means. Fisher counts below use the direct per-seed comparison, not a fitted Fisher model.

| Policy | LOSO scalar histogram ≤25% | Full Fisher ≤25% (descriptive) |
|:--|--:|--:|
| balanced_block | 0/5 | 1/5 |
| alternating | 1/5 | 3/5 |

The scalar baseline tests this specific count-energy explanation. Its result does not rule out richer uses of histogram information or establish data sufficiency. Policy-order screens are descriptive and do not enter the verdict unless a frozen theory predicts that ordering.

The analysis is limited to this finite transfer task on the evaluated frozen models. It establishes no NAACL novelty, generalization beyond this setup, or formal theorem.
