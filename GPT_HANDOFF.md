# Incremental review: completed Round 6 toy experiments

Review base: `8f944c69c6359d023cb1322563f17f64149a9f3c`.
Evidence head: `f9d79b784ae40573a0ded7897ca5abf605d85fda`.
This handoff is a later metadata-only commit; the evidence head stays fixed.

## Minimal reading order

1. [RESULTS.md](RESULTS.md), the Round 6 section before the historical E0 result.
2. [Completed derived report](runs/round6_toy_v1_20261010T061030Z/snapshots/20261010T073818Z/analysis/RESULTS.md)
   and its three figure pairs. Do not start with the large JSON or JSONL files.
3. [Frozen protocol](experiments/PROTOCOL_ROUND6_TOY_V1.md) and
   [GPT_CONTEXT.md](GPT_CONTEXT.md) for exact source symbols and estimands.
4. [THEORY_STATUS.md](THEORY_STATUS.md) before assessing the new
   [proof draft](THEOREM_AND_PROOF.md).

## Changed evidence and claims

The earlier review base concerned Experiment 0. The evidence head adds the
Round 6 protocol, code, tests, proof draft and completed analytic/trained study:

- Experiment 1: 160 exact configurations, 40 sequence enumerations, all 12
  recorded gates pass. It supports the specified propagation-scale construction.
- Experiment 2: 20 models, 22,080 prompt rows and 276 complete groups. All five
  training seeds per task/depth are retained in the primary aggregates.
- Histogram passed its frozen quality gate for 10/10 models. Its measured
  horizon curves slow, and its rho-mixture KL ratios approximately follow rho
  squared. Mixtures store n+m records and are precision diagnostics.
- Recall passed for only 3/10 models (1/5 depth 2, 2/5 depth 4). The seven
  failures remain visible; this is not robust evidence from learned recall.
- The coarse global certificate is no tighter than the model-specific
  universal bounded-logit cap in 840/840 applicable group-seed cases. No fitted
  constant or uniform-query empirical certificate is reported.

The completed endpoint is
`runs/round6_toy_v1_20261010T061030Z/snapshots/20261010T073818Z/`.
The run's top-level `analysis/` is a preserved earlier Experiment-1-only
snapshot, not the completed endpoint. Snapshot provenance binds frozen source,
configuration and the public endpoint files; `publication_manifest.json`
binds the completed derived analysis files and sanitized execution times.

## Unchanged claims and unresolved issues

Experiment 0's full result is preserved: `NO_HELDOUT_SUPPORT`, all 144 cases,
prediction-MSE reduction -0.184% with an interval crossing zero. Its protocol,
data, code and numerical evidence are unchanged.

The new study does not establish a general trained-Transformer theorem,
nonvacuous uniform-generation certificate, minimax cache-space lower bound,
fixed-memory streaming, sequence-TV endpoint, practical inference advantage
or publication-level novelty. The suffix remains exact and grows with T.
Historical proof-verification and novelty files referenced by the proof draft
are absent and have not been recreated by these narrower experiment checks.

## Verification actually performed

All 12 Round 6 unit tests, the exporter test and two E0 analysis tests pass.
`python -B scripts/verify_round6_release.py` verifies 12 source hashes, three
public payload hashes, eight derived-analysis hashes, seed/group coverage,
endpoint-to-group aggregation, 2,880 full-cache identity rows and recomputes
the 160-row analytic grid. The E0 offline verifier still returns
`NO_HELDOUT_SUPPORT`. These are software/artifact checks, not a scientific pass.
Checkpoint norms were not recalculated during publication.

## Concrete reviewer questions

1. Does one-time prefix replacement plus exact self-generated suffix append,
   including the CONT boundary, implement the proof's timing and KL estimand?
2. Which trained histogram findings go beyond analytic attention dilution,
   given sampled-query errors and the uninformative global sufficient bound?
3. Is further learned-model evidence interpretable before a new frozen recall
   training protocol reliably passes its task gate across seeds?
