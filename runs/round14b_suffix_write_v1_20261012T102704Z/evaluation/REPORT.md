# Round 14B terminal scientific report

Verdict: **TASK_OR_INTERVENTION_UNQUALIFIED**.

Same-seed prerequisites: 2/5; all five gates: 0/5 (required 4/5).

All fixed seeds are retained. Model seeds are the independent units; other repetitions are nested.

| Seed | TF answer | P answer | P syntax | P query | Cache-delete drop | Task | Memory | Matched | Risk | Fisher |
|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| 11 | 0.762 | 0.756 | 0.998 | 0.996 | 0.549 | FAIL | PASS | FAIL | PASS | FAIL |
| 23 | 1.000 | 0.999 | 1.000 | 0.999 | 0.770 | PASS | PASS | PASS | PASS | FAIL |
| 37 | 0.863 | 0.859 | 0.999 | 0.999 | 0.592 | FAIL | PASS | FAIL | PASS | FAIL |
| 51 | 0.838 | 0.837 | 1.000 | 0.999 | 0.609 | FAIL | PASS | FAIL | PASS | FAIL |
| 71 | 1.000 | 1.000 | 1.000 | 1.000 | 0.766 | PASS | PASS | PASS | PASS | FAIL |

| Policy | KL(Q||P) | Full Fisher | Diagonal Fisher | Mean per-seed relative error |
|---|---:|---:|---:|---:|
| balanced_block | 49.8354 | 4.0223 | 3.88952 | 90.261% |
| alternating | 50.1825 | 3.86895 | 3.89441 | 90.988% |

## One-shot Round 7 control, n=90, m=45, T=94

The control applies one original-record summary per layer, then appends each stream's own KV. Continuous compression is a different intervention.

| Method | KL(Q||P) | KL(P||Q) |
|---|---:|---:|
| full | 0 | 0 |
| gs_thin | 11.7477 | 6.83986 |
| uniform_subset | 15.769 | 6.48423 |

See summary.json for every seed and cell ranges; continuous.jsonl and one_shot.jsonl contain within-cell Monte Carlo standard errors. NPZ traces are secondary evidence.

Theta=.25 is local diagnostic evidence only. The endpoint has 16 fewer records out of 184 (8.70%), not half of the entire cache. The one-shot control halves the 90-record prefix, not future records.

Retrospective Q-history diagnostic; one-shot descriptive control; no certificate, inference speedup or optimal-space claim.

## Closure

Final fixed run; no rescue seeds, thresholds, sweeps or follow-up round.
