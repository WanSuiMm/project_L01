# Round 13 forward-KL and Fisher probe

Execution: **FULL_RUN**; coverage complete: **True** (640/640 rows).

The independent unit is the frozen model seed. Rows are averaged equally over the two phases and prompts within each seed.
At amplitude 1, forward replay KL is the actual sequence risk under Q1 histories; full and diagonal Fisher predictions use those same histories.
The amplitude 0.25 rows replay the same Q1 histories and are local diagnostics, not Q0.25 sequence risk. Predictions already include amplitude squared.

Endpoint horizon: 129; relative-error threshold: 0.25; required seeds: 4.

The Round 12 reference is `round12_all_paths_endpoint_kl`: the original 64-path, two-phase-averaged T=129 endpoint. Its duplicated rows are reduced once per seed/prompt/policy, then prompts and seeds are averaged equally.

| Policy | Selected subset mean (configured endpoint) | Original 64-path mean (T=129 source) |
|:--|--:|--:|
| balanced_block | 1.21654 | 1.20193 |
| alternating | 0.0997674 | 0.107638 |

| Policy | Mean actual | Mean full | Mean diagonal | Full rel. error | Diagonal rel. error | Within screen | Full better than diagonal |
|:--|--:|--:|--:|--:|--:|--:|--:|
| balanced_block | 1.21654 | 1.22009 | 0.178589 | 0.00322204 | 0.853192 | 5/5 (PASS) | 5/5 (PASS) |
| alternating | 0.0997674 | 0.0997077 | 0.18148 | 0.00171461 | 0.823534 | 5/5 (PASS) | 5/5 (PASS) |

| Block / alternating endpoint-risk ratio | Actual | Full Fisher | Diagonal Fisher |
|:--|--:|--:|--:|
| Ratio of equal-seed policy means | 12.1937 | 12.2367 | 0.984066 |

Amplitude 0.25 local diagnostic (equal phase/prompt/seed aggregation):

| Policy | Replay KL on Q1 histories | Full prediction | Diagonal prediction | Vector remainder |
|:--|--:|--:|--:|--:|
| balanced_block | 0.0761868 | 0.0762557 | 0.0111618 | 0.0203981 |
| alternating | 0.00623222 | 0.00623173 | 0.0113425 | 0.00728618 |

These are endpoint-law-weighted diagnostics, not the P-law sequence Fisher at zero and not held-out prediction. They establish no causal, theorem, or certification claim.

Figures: `analysis/round13_endpoint.png` and `analysis/round13_prefix.png`.
