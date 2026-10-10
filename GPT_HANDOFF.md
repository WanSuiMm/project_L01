# Incremental review: completed Round 7 thinning study

Review base: `d53807c9752a13a8c59b029157c9137bfe46e0cd`.
Evidence head: `05af15a6d9f8f43736422364fbc7de4252a63c6e`.
This later commit changes only this handoff; the evidence head stays fixed.

## Minimal reading order

1. [RESULTS.md](RESULTS.md), the Round 7 section before previous Round 6/E0.
2. [Completed generated report](runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/analysis/RESULTS.md)
   and its budget, horizon and multilayer figures. Do not start with raw logs
   or the 1.1 MB aggregate JSON.
3. [Frozen protocol](experiments/PROTOCOL_ROUND7_THINNING_V1.md) and
   [GPT_CONTEXT.md](GPT_CONTEXT.md) for both KL estimands and source symbols.
4. [THEORY_STATUS.md](THEORY_STATUS.md) before the
   [Round 7 proof snapshot](theory/archive/round7/THEOREM_AND_PROOF.md).

## Changed evidence and claims

Round 7 adds an exact equal-weight subset construction and a completed
GS-Thin versus uniform-subset evaluation on the ten qualified Round 6
histogram checkpoints. No new training or context-table extension was performed.

- All 70,560 trained records pass server validation; 252 groups and 108 paired
  comparison cells retain all five training seeds per depth.
- GS-Thin has lower mean sequence KL in 96/108 cells. All 12 exceptions use
  n=16,m=2: six depth-4 intervals favor uniform, and six depth-2 intervals cross
  zero. The 108 seed-bootstrap intervals are unadjusted for multiplicity.
- KL/h is nearly constant over the tested horizons, h=nT/(n+T). Its max/min
  ratio across horizons has median 1.059 for GS, largest 1.205. This is finite
  histogram evidence, not asymptotic saturation or a trained-model m^-2 theorem.
- Exact hard-class minima through T=128, one exact witness through T=2048, and
  balanced-uniform expected risk are distinct analytic endpoints. Long-horizon
  class/control aggregates are curvature envelopes, not exact measured KL.
- The global forward small-risk condition holds in **0/17,280** GS Q-to-P
  rows. log V is 161.01–370.34. This remains the main negative result;
  empirical MMD does not certify the theorem interface.

The compact completed endpoint is
`runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/`.
Its publication manifest binds 19 frozen source/configuration files and 17
public payloads. Raw trained JSONL, weights and operational records are excluded.
Trained replay requires the original Round 6 checkpoints; their hashes,
quality records and norm audits are public. The archived theory files are
preserved as supplied; their historical novelty/verification wording is qualified.

## Unchanged evidence and claim limits

Round 6 numerical evidence and code are unchanged, including seven failed
recall training seeds, 840/840 uninformative global bounds, and the distinct
rho-mixture precision diagnostic. Experiment 0 remains NO_HELDOUT_SUPPORT,
144 cases, prediction-MSE reduction -0.184% with an interval crossing zero.

The Round 7 lower-bound class is equal-weight subsets of original records.
It does not cover arbitrary positive weights, synthetic representatives or
arbitrary encoders. The imported SG theorem is not independently certified
for the documented float64 implementation and simultaneous-boundary rule.
The suffix grows with T. No uniform-generation certificate, practical
inference/memory gain, general NLP validation or publication readiness follows.

## Verification actually performed

Eight GS-Thin tests and two exact directional-law/identity tests pass.
`python -B scripts/verify_round7_release.py` passes 114 checks, including
source/payload hashes, checkpoint metadata, 252 primary seed-level means/CIs,
108 paired differences/CIs and analytic provenance/output hashes. It does
not reconstruct the omitted raw-row keys, rerun the analytic DP, load weights,
or recompute checkpoint norms. Raw completeness is recorded server validation.
The Round 6 release verifier was rerun and passes; these are artifact/software
checks, not independent theorem verification.

## Concrete reviewer questions

1. Does the imported SG interface cover the actual recursive float64 sampler,
   including the explicitly documented simultaneous-boundary handling?
2. How much of the finite trained histogram result goes beyond attention
   dilution, given the smallest-budget losses and vacuous global constants?
3. Is the restricted class lower bound useful enough while arbitrary-weight
   summaries and nonvacuous model-dependent guarantees remain unresolved?
