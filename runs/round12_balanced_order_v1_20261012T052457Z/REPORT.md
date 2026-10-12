# Round 12 analysis

Execution: **FULL_RUN**; prefix coverage: **COMPLETE**; overall coverage: **COMPLETE**.

The configured design contains 5 model seeds, 4 prompts, and every prefix horizon 1–129. Expected prefix rows: 10,320; observed: 10,320.

Within each model seed, the two schedule phases are averaged equally for each prompt and horizon, then prompts are averaged equally. Model seeds are the independent units. `sequence_kl_path_se` is a path-level Monte Carlo quantity and is not seed uncertainty.

Terminal KL order: balanced_block > alternating in 5/5 assessed seeds; 4 are required by config. This screen uses order count only and has no effect-size cutoff.
Terminal local comparability: 5/5 assessed seeds satisfy max/min local-attention squared error ≤ 1.25.

| Model seed | KL block/alt | Block KL > alt | Local max/min | Local comparable | Token-bias² block/alt | Cache-bias² block/alt |
|---:|---:|:---:|---:|:---:|---:|---:|
| 11 | 11.1951 | True | 1.00462 | True | 1.20542 | 1.47384 |
| 23 | 10.6817 | True | 1.0043 | True | 0.916082 | 1.29236 |
| 37 | 11.2878 | True | 1.00194 | True | 1.01078 | 1.42365 |
| 51 | 12.9399 | True | 1.00341 | True | 1.17587 | 1.64916 |
| 71 | 10.1531 | True | 1.00404 | True | 0.902775 | 1.30759 |

Integrated prefix bias energy is the sum over horizons of the already phase-averaged and prompt-averaged squared-bias means:

| Model seed | Policy | Token-bias² sum | Cache-bias² sum |
|---:|:---|---:|---:|
| 11 | balanced_block | 1.25327 | 195.621 |
| 11 | alternating | 0.105938 | 30.4938 |
| 23 | balanced_block | 1.22245 | 191.871 |
| 23 | alternating | 0.111452 | 30.2849 |
| 37 | balanced_block | 1.25249 | 195.403 |
| 37 | alternating | 0.10957 | 28.3592 |
| 51 | balanced_block | 1.28287 | 234.972 |
| 51 | alternating | 0.104003 | 28.6085 |
| 71 | balanced_block | 1.29377 | 197.905 |
| 71 | alternating | 0.119881 | 33.8713 |

Optional terminal path file: **COMPLETE**.

Figures:
- `analysis/sequence_kl_horizon.png`
- `analysis/bias_squared_trajectories.png`
- `analysis/local_perturbation_vs_cache_moments.png`

Interpretation boundary: this is a finite frozen-checkpoint comparison. It supports no causal attribution, no no-bias claim, no asymptotic-rate or certification claim, and no Fisher or novelty claim.
