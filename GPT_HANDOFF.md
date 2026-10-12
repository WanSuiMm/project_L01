# Incremental review: Round 14B final run and one-shot control

Review base: `7f59ab24f481cde4bf4cbd4d2644e303e96c2025` (published Round 14).
Evidence head: `c962d66335517a33aae22bd5ae9687e0fc9fc8a7`.
This later metadata-only commit updates this handoff; the evidence head stays fixed.

## Minimal reading order

1. [RESULTS.md](RESULTS.md), Round 14B only: frozen gates, endpoint and one-shot comparison.
2. [Interpretation](runs/round14b_suffix_write_v1_20261012T102704Z/INTERPRETATION.md), then the [report](runs/round14b_suffix_write_v1_20261012T102704Z/evaluation/REPORT.md) and [summary](runs/round14b_suffix_write_v1_20261012T102704Z/evaluation/summary.json).
3. [Protocol](experiments/PROTOCOL_ROUND14B_SUFFIX_WRITE_V1.md), [configuration](experiments/configs/round14b_suffix_write_v1.json), and the new Round 14B section of [GPT_CONTEXT.md](GPT_CONTEXT.md).
4. [Record verification](runs/round14b_suffix_write_v1_20261012T102704Z/release_verification.json) and [terminal closure](runs/round14b_suffix_write_v1_20261012T102704Z/CLOSURE.md).

The 270 NPZs and 1,040 primary JSONL rows are secondary evidence. Earlier
Round 6–14 records and theory do not need rereading for the empirical update.
The [novelty audit](NOVELTY_AUDIT_2026-10-12.md) and
[pre-run protocol review](ROUND14B_PROTOCOL_REVIEW_2026-10-12.md) are separately
labeled historical research decisions, not completed experiment reports.

## New evidence and narrower claim

The final suffix-write task uses five new two-layer width-128 models trained
5,000 steps. The initial prompt has no values; fresh random value pairs are
written before delayed READs. Same-role value KV is edited after vb's own
forward pass. Plus/minus selects first/second records independently of token
IDs. Both balanced antipodes of Block and Alternating are evaluated at theta=1
and a separate theta=.25 diagnostic. Task, selector and event timing differ
from Round 14, so this does not causally isolate task identity alone.

The run completed in 666.17 seconds at 10:39:23 UTC on 2026-10-12, with
320 continuous rows, 720 one-shot rows, 10 P-quality rows, five cache-deletion
rows and 270 NPZs. Cache deletion preserves all inputs/queries/truth and
recomputes future KV. Accuracy drops 54.88–76.95 percentage points; memory
necessity and material sequence risk pass all seeds.

Only seeds 23 and 71 pass all four prerequisites; 0/5 pass all five gates
(requires the SAME >=4/5). Verdict: **TASK_OR_INTERVENTION_UNQUALIFIED**.
No threshold was relaxed and no failed seed was removed or retrained.
All-five Block/Alternating KL is 49.8354/50.1825, Full Fisher 4.02230/3.86895,
diagonal 3.88952/3.89441. Full relative errors average 90.26%/90.99% and fail
the <=25% condition in every seed. Both qualified models underestimate by >99%,
so training failure alone cannot explain the endpoint. Block exceeds
Alternating in only 2/5 seeds; no consistent ordering benefit is established.

The one-shot control applies actual GS/uniform original-record summaries
once independently per layer to exact n=90 prefix KV, m=45, then appends
each model's own suffix KV for T=94. Each KL direction uses its appropriate
sampled law plus the other shadow. GS/uniform KL(Q||P)=11.7477/15.7690,
KL(P||Q)=6.83986/6.48423, with seed wins 3/5 and 2/5. Full identity is zero.
This descriptive control does not establish a useful bound, space rate or
refutation of the conditional Round 7 theorem. Continuous saves 16/184 total
records; one-shot saves 45/184. Their risks cannot isolate timing at a matched
total budget. Theta=.25 has 200 records and cannot rescue the physical endpoint.

## Validation and unchanged boundaries

Eleven focused tests and local/server end-to-end smokes passed before dispatch.
Saved-record verification checks 14 source hashes, config binding, coverage,
uniqueness, KL/Fisher trace sums, event-Gram sum/trace, mass/cardinality,
original subsets, and raw seed-gate arithmetic:

```text
python -B scripts/verify_round14b_release.py --input-dir runs/round14b_suffix_write_v1_20261012T102704Z
```

It does not load absent checkpoint bytes, regenerate tokens, replay JVPs or
verify a theorem. An independent raw-record review found no concrete
invalidating data/invariant mismatch. Weights, private receipts, machine
identifiers, transient logs and unrelated archives are excluded.

Round 13's finite histogram diagnostic, Round 14's negative transfer, and
Round 9's certificate failures are unchanged. Fisher uses conditional P
geometry on evaluated Q histories, not P-law sequence Fisher, heldout
forecasting or a finite remainder bound. No useful certificate, general
compressor advantage, inference speedup, optimal space, global novelty or
publication readiness is established. This final run is closed without rescue
sweeps or a new round.

## Reviewer questions

1. Are prerequisites and the failed explanation gate separated clearly, especially for qualified seeds 23 and 71?
2. Do trace sums and Gram identities support the arithmetic without being mistaken for a remainder guarantee?
3. Does one-shot preserve exact-prefill, independent layer summaries, own-suffix KV and both KL sampling directions?
4. Are different budgets, new event timing, secondary theta=.25 and unchanged conditional Round 7 status explicit?
