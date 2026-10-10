# Round 7 thinning v1: completed

Current status: **COMPLETE / ANALYTIC_GATES_PASS / HISTOGRAM_QUALIFIED /
GLOBAL_SMALL_RISK_UNCERTIFIED**. This is a completed result snapshot, not a monitor.

Completed: 2026-10-10 11:21:13 UTC (19:21:13 Asia/Shanghai).
Elapsed: 2,776.51 seconds. All three stages exited zero. The trained stage
completed 70,560/70,560 records and 10/10 models with no new training.
GS-Thin has lower cell-mean KL in 96/108 comparisons; all 12 exceptions use
n=16,m=2. No forward GS row passes the audited small-risk condition (0/17,280).
The five checkpoint seeds are the independent units, not the grid cells.

Read [canonical interpretation](../../RESULTS.md), then the
[completed report](snapshots/20261010T115125Z/analysis/RESULTS.md) and
[publication manifest](snapshots/20261010T115125Z/publication_manifest.json).
The compact snapshot preserves analytic CSVs, all trained seed-level aggregate
values and figures. Raw trained rows, weights and operational records are
excluded. Public verification does not reconstruct raw-row validation.

## Preserved dispatch record

Status at dispatch: **RUNNING / ANALYTIC**. This is a launch record, not a live
monitor or a completed scientific result.

Run ID: `round7_thinning_v1_20261010T102523Z`.
Launch: 2026-10-10 10:34:56 UTC (18:34:56 Asia/Shanghai).
The registered primary server runs the CPU thinning construction and GPU 1
decoder evaluation. Host paths, process IDs, logs and operational receipts are
kept in the private run directory. All 19 deployed file hashes matched the
frozen local source snapshot.

## Frozen question and design

Evaluate one-time, mass-preserving prefix compression and both directional
per-summary sequence KL risks. Test the restricted worst-case lower bound on
analytic histogram instances; measure trained-model behavior separately.

- Analytic: 18 `(n,m)` pairs; horizons 8, 32, 128, 512, 2048. Enumerate all
  feasible equal-weight summary counts through horizon 128. Compute exact
  all-horizon DP for one minimum-gap witness. Longer balanced-uniform controls
  retain explicitly labeled curvature envelopes.
- Trained: ten existing qualified width-64 histogram models, depths 2/4 and
  training seeds 11/23/37/51/71. No new training or context-table extension.
- Prefix lengths 16/64/128; horizons 8/32/128; record fractions 1/8, 1/4, 1/2.
- GS-Thin and uniform-subset interventions; eight compression seeds, eight
  held-out prefixes per length, and 16 sampled paths per direction. Full-cache
  identity controls are run once per prefix.
- Expected trained endpoint records: **70,560** (1,440 full, 34,560 GS-Thin,
  34,560 uniform). Uncertainty resamples training seeds after nested averaging.

## Verification before dispatch

Eight GS-Thin tests and two directional-law tests passed. The latter compare
Monte Carlo chain-rule estimates with exact short-sequence enumeration and
check full-cache identity. The server smoke completed all six expected records;
both full-cache KL directions and the identity logit difference were zero.
Report generation passed on those real smoke records in an isolated fixture.
The analytic qualification passed all recorded gates. These are software and
analytic checks, not GS-Thin empirical performance results on the formal grid.

The float64 implementation follows recursive GS halving, uses minimum-norm
least squares for singular Grams, and explicitly freezes all simultaneous box
boundary hits. Numerical PSD corrections and rank tolerances are documented
in the protocol. Empirical tests do not certify the imported SG theorem.

## Outputs and interpretation

The durable supervisor runs `round7_analytic.py`, `round7_transformer.py`, then
`analyze_round7.py`, with a 43,200-second total limit. New outputs are routed to
`analytic/`, `transformer/`, and `analysis/` under this run. The analysis writes
seed-level aggregates, paired comparisons, three PNG/PDF figure pairs and a
result summary, and returns failure on incomplete or invalid evidence.

No trained-model scaling exponent is a pass/fail criterion. The GS-only global
risk diagnostic uses fixed model norms, separately from observed KL and MMD.
An uninformative certificate is preserved. The cache budget is for the old
prefix; each model still appends `T` suffix records. No inference speed claim
is made.

Protocol: [PROTOCOL_ROUND7_THINNING_V1.md](../../experiments/PROTOCOL_ROUND7_THINNING_V1.md).
Source: [round7_transformer.py](../../experiments/round7_transformer.py),
[gs_thin.py](../../experiments/gs_thin.py),
[round7_analytic.py](../../experiments/round7_analytic.py),
[analyze_round7.py](../../experiments/analyze_round7.py).
