# Incremental review: Round 14 associative-recall transfer

Review base: `4de82c6e298821499777eea3203f671b0ac9b46c` (published Round 12/13).
Evidence head: `658a534f32625025f4f14d88b0a7e5d6c1a12856`.
This later commit changes only this handoff; the evidence head stays fixed.

## Minimal reading order

1. [RESULTS.md](RESULTS.md), Round 14 only: qualification, primary screens and limits.
2. [Round 14 interpretation](runs/round14_recall_transfer_review_v1_20261012T085656Z/INTERPRETATION.md), then the [frozen report](runs/round14_recall_transfer_v1_20261012T083530Z/evaluation/analysis_round14/REPORT.md) and [summary](runs/round14_recall_transfer_v1_20261012T083530Z/evaluation/analysis_round14/summary.json).
3. [Frozen protocol](experiments/PROTOCOL_ROUND14_RECALL_TRANSFER_V1.md), [configuration](experiments/configs/round14_recall_transfer_v1.json) and the new Round 14 section of [GPT_CONTEXT.md](GPT_CONTEXT.md).
4. [Recorded-evidence verification](runs/round14_recall_transfer_review_v1_20261012T085656Z/release_verification.json); inspect code only for the concrete questions below.

Raw trajectories and large prefix/event CSVs are secondary. Round 11--13
records and theory do not need rereading for this incremental update.

## New evidence and narrower claim

One new fully generative eight-key associative-recall task adapts the key/value
retrieval idea; it does not reproduce the masked-loss MQAR benchmark. Five
new width-128, two-layer models trained for 5,000 steps each. Evaluation uses
two heldout n=16 tables, T=129, 16 Q paths per antipodal phase, and the original
Balanced Block/Alternating thinning and full-unroll event-JVP machinery.

The run completed in 780.68 seconds with 160 metric rows, 10 P-quality rows,
40 Q-quality rows and 30 NPZs. Only 4/5 models pass training and autonomous
generation gates. Seed 37 has all-answer/first-answer top-1 42.49%/43.55%
and P-generated answer accuracy 36.21%. The frozen all-five requirement gives
**TASK_UNQUALIFIED**; no seed was removed, replaced or retrained.

All-five equal-seed KL(Q1||P) is 0.000668265/0.000274418 (Block/Alternating);
full Fisher is 0.000567869/0.000246408; diagonal 0.000302254/0.000301715.
Full error <=25% passes 1/5 and 3/5 (needs >=4/5 each); full beats diagonal
in 4/5 and 2/5. Both policy-level transfer screens fail. Seed 37 contributes
79.4%/71.0% of pooled KL and is the sole Block accuracy pass. An exploratory
qualified-four sensitivity still fails, so the failed training seed alone
does not rescue the Fisher approximation.

Block KL remains larger in all five seeds. Local-attention mean-error
comparability passes only 1/5. The LOSO scalar histogram-energy baseline
also fails (0/5 and 1/5 accuracy passes), which does not establish a unique
mechanism. The retrieval table remains exact and learned models can ignore
redundant suffix pairs; tiny KL and unchanged sampled Q answers expose this
intervention's limited strength. Task, width and prefix length all differ
from Round 13, so the transfer is not an isolated task-identity intervention.

## Validation and unchanged boundaries

Two task-definition tests pass. Local/server interface smokes passed before
formal execution. Saved-record verification checks coverage, source hashes,
configuration binding, checkpoint-summary identity, trajectory/CSV arithmetic,
full-horizon Gram sum/trace and seed screens. It does not load model weights,
rehash their absent bytes, regenerate tokens or re-evaluate JVP correctness.
The frozen summary's `run_id=evaluation` labels the analyzer's input directory;
the scientific parent run identity is established by its source/provenance.

Round 13's successful finite histogram diagnostic stands unchanged. Its
quantitative transfer is not established here. Fisher still uses conditional
P-logit geometry averaged on evaluated Q1 histories; it is not P-law sequence
Fisher, heldout forecasting, unique mediation or certification. Earlier
certificate failures and all other rounds remain intact. No whole-cache
half-budget, practical speedup, new theorem or NAACL novelty follows.
No additional experiment or tuned rescue was launched.

## Reviewer questions

1. Does the seed-level negative endpoint remain clear despite the much smaller pooled approximation error?
2. Does the exact retrieval table and redundant compressed suffix limit this experiment's ability to probe output-relevant memory?
3. Are task-qualification failure, local-error mismatch and higher-order approximation error separated adequately, without attributing the failure to one untested cause?
4. What single controlled comparison would most sharply test the remaining finite mechanism claim? The repository does not claim that such a follow-up has run.
