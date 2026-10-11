# Incremental review: Round 9 local-bound endpoint

Review base: `7558461177239a910d9196691fd74d174442a07c`.
Evidence head: `bd5b6903249307cce2cd13671c232e5ba8a23bf3`.
This later commit changes only this handoff; the evidence head stays fixed.

## Minimal reading order

1. [RESULTS.md](RESULTS.md), Round 9 section, for the endpoint and
   claim boundary.
2. [Run status](runs/round9_local_cert_v1_20261011T045942Z/RUN_STATUS.md)
   for dispatch and retrieval, followed by the public snapshot at
   `runs/round9_local_cert_v1_20261011T045942Z/snapshots/20261011T174341Z/`.
   `summary.json` is the aggregate entry point.
3. `verification.json` and `condition_metrics.csv` for recorded checks and
   condition-level values. `checkpoints.json`, `config.json`, `prefixes.json`
   and `provenance.json` provide the frozen bindings. The snapshot's generated
   `RESULTS.md` still carries its historical Round 6 title; the top-level
   Round 9 section is the current project-facing route.
4. Source map: `experiments/round9_original_models.py:load_original`,
   `reference_tree`, `make_summary`, `evaluate_summary`, `aggregate`;
   `experiments/round9_certify.py:bound_all_steps`,
   `certified_conditional_bound`; and
   `scripts/analyze_round9_endpoint.py:validate`, `load_rows`.

## Changed evidence and claims

Round 9 completes 330/330 finite-horizon local-bound conditions on the five
original qualified width-64, two-layer Round 6 histogram models. No training
was performed. At n=64,m=32, agglomerative has mean upper KL(Q||P) 8.4763
(range 0.919 to 36.623), with 0/15 prefixes <=0.1, 1/15 <1 and 15/15 below the
reference cap. GS-Thin and uniform subset both have mean upper 37.4676; each
has 0/15 <=0.1 and 9/15 below the cap. Each method passes the frozen seed gate
in 0/5 models; the requirement is 4/5, with at least two of three prefixes
<=0.1 per model.

The near-full agglomerative control at m=63 has mean upper 0.0065908 and
15/15 <=0.1, but retains 63/64 records. At m=48, agglomerative has mean upper
4.1374, 10/15 <1 and 0/15 <=0.1. These are finite float64 analytic
evaluations, not strict interval certificates. Falling below the reference
cap is distinct from meeting the high-fidelity gate. No memory or inference
speed advantage was measured.

Round 6 and Round 7 evidence and claims are unchanged. Keep the conclusion
narrow to the frozen Round 9 intervention and numerical gate: it does not
falsify a theorem, establish a novelty claim, or support a general Transformer
guarantee.

## Verification actually recorded

Eight Round 9 local software tests pass. Four server numerical/software tests
and the 11-condition original-model smoke passed before the formal run.
The compact endpoint checker passes on both the retrieved rows and the public
CSV projection; these checks are separate from the failed scientific gate.

`verification.json` reports `RECORDED_ENDPOINT_CHECKS_PASS` for 330 unique
expected conditions, frozen source/config/checkpoint metadata, 40 histories
and 81 leaves per condition, recorded sequence inequalities, full-cache
identities, aggregate arithmetic and the predeclared gate. The export verifier
does not replay model weights or the omitted per-history inequalities, and it
does not construct an outward-rounded interval enclosure. The endpoint status
and numerical gate outcome are separate: execution is complete, while all
three 50% budget feasibility gates fail.

## Concrete reviewer questions

1. Do the large radii and tube fractions at m=32, with the sparse agglomerative
   final-LayerNorm zero-variance intersections, adequately describe where this
   finite certificate loses tightness without implying a causal diagnosis?
2. Is the public claim limited to the failed 50% numerical-feasibility gate and
   the narrow near-full comparison, without promoting below-cap counts into a
   high-fidelity result?
