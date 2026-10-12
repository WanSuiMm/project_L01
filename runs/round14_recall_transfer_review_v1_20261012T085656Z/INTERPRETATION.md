# Round 14: completed transfer attempt, negative frozen endpoint

Evidence: `round14_recall_transfer_v1_20261012T083530Z`, completed
2026-10-12 08:50:22 UTC in 780.68 whole-pipeline seconds. All five seeds
completed 5,000 steps. The 160 metrics, 10 P-quality records, 40 Q-quality
records and 30 NPZ files are retained. Frozen outputs were not rewritten.

The question was whether Round 13's quantitative same-history cross-event
explanation survives a new fully generative retrieval task, and whether it
adds to a simple cumulative count-energy explanation. The frozen all-five
qualification fails and neither policy passes both Fisher screens.

## Qualification

| Seed | Validation all-answer top-1 | First-answer top-1 | Unmasked P answer accuracy | Qualified |
|---|---:|---:|---:|---|
| 11 | 100% | 100% | 99.9756% | yes |
| 23 | 100% | 100% | 99.9023% | yes |
| 37 | 42.4850% | 43.5547% | 36.2061% | no |
| 51 | 100% | 100% | 100% | yes |
| 71 | 100% | 100% | 100% | yes |

All seeds pass the P-query-validity >=95% threshold. Training requires both
top-1 metrics >=90%; P answer sampling requires >=80% over every answer
position, including invalid-query cases. Five qualified seeds were required.
The official verdict is **TASK_UNQUALIFIED**. No failed seed was replaced.

## Quantitative transfer

| T=129, all-five equal-seed mean or screen count | Balanced Block | Alternating |
|---|---:|---:|
| Actual KL(Q1||P) estimate | 0.000668265 | 0.000274418 |
| Full Fisher | 0.000567869 | 0.000246408 |
| Diagonal Fisher | 0.000302254 | 0.000301715 |
| Mean per-seed full relative error | 37.27% | 21.29% |
| Full error <=25% | 1/5 | 3/5 |
| Full closer than diagonal | 4/5 | 2/5 |
| LOSO scalar histogram error <=25%, descriptive | 0/5 | 1/5 |

Each primary screen needs >=4/5 seeds for each policy. Block fails accuracy;
Alternating fails accuracy and comparison against diagonal. The small pooled
relative error must not replace the frozen per-seed screen: seed 37 contributes
79.4% of Block and 71.0% of Alternating KL and is the only Block accuracy pass.
Seed 23 is slightly outside the Block threshold (25.1764% error), while other
qualified Block seeds have appreciably larger errors.

An exploratory sensitivity retaining only the four qualified seeds gives
Block accuracy 0/4 and full-versus-diagonal 3/4; Alternating gives 2/4 and 1/4.
This is not a new formal endpoint or permission to discard a failed seed. It
shows that the training failure alone cannot rescue the Fisher approximation.

Block KL exceeds Alternating KL in all five seeds. Among the four qualified
models the ratios are 1.45--1.87, whereas the corresponding count-energy
ratios are 4.13--4.50. However, the local mean-attention-error max/min screen
passes only 1/5 seeds (ratios 1.19--2.01). The contrast is not a controlled
demonstration that temporal order alone causes the KL gap. The token-ID
selector also aligns with query/answer roles on valid suffix pairs.

The per-cell full-Fisher relative-error mean at T=129 is 33.75%/21.25%; the
maxima are 79.61%/54.38%. Aggregate agreement does not erase those residuals.
Sampled Q answers remain correct in all evaluated paths of the four qualified
models: these tiny KL changes did not produce observed task-accuracy loss.
This is a finite sample, not an accuracy guarantee.

The retrieval table itself remains exact. A model that has learned lookup
can answer from that table while ignoring redundant generated suffix pairs.
Thus the tiny KL values also expose a limited intervention: this experiment
does not test compression of the task's essential prefix memory. It cannot
establish a failure of Fisher geometry for all retrieval compression regimes.

## What this changes

Round 13's successful finite histogram result remains intact. Its unusually
accurate quadratic fit has not transferred quantitatively in this realization.
The scalar histogram baseline is also inadequate; that does not establish
Fisher as a unique mechanism or rule out richer count-dependent explanations.

The Fisher diagnostic uses conditional P probabilities on evaluated Q1
histories. It is not heldout forecasting, P-law sequence Fisher, a sequence-KL
Taylor coefficient at zero, or a rigorous bound. Task, model width and prefix
length differ from Round 13, so this is one bundled task-transfer attempt,
not an isolated causal intervention on task identity. No novelty verdict,
general Transformer theorem, full-cache memory reduction or speedup follows.

No follow-up training or tuned rescue was launched. The defensible current
claim stays restricted to the earlier histogram setting. Read the
[frozen report](../round14_recall_transfer_v1_20261012T083530Z/evaluation/analysis_round14/REPORT.md)
and [verification](release_verification.json) before secondary raw trajectories.
Record verification does not load model weights or re-evaluate derivatives.
The frozen analysis summary labels its input directory as `run_id=evaluation`;
the scientific run identity is the parent directory named at the top, bound by
the source/config provenance. This label is preserved rather than rewritten.
