# Round 14B: final interpretation and scope of closure

The final fixed run completed in 666.17 seconds on 2026-10-12 at 10:39:23 UTC.
The frozen verdict is **TASK_OR_INTERVENTION_UNQUALIFIED**, with prerequisites
passing in 2/5 seeds and all five gates passing in 0/5. Execution completed;
the scientific endpoint did not pass. No new round or rescue tuning follows.

## What the intervention fixed

The initial prompt contains no value tokens. Compression acts on adjacent
same-role values after their own forward computation, with first/second
selection independent of token IDs. Both signs and antipodes have equal event
counts, mass and physical budgets. Cache-only deletion preserves all tokens,
queries and truth while future KV is recomputed inside each stream.

That deletion reduces heldout answer accuracy by 54.88–76.95 percentage points
in every seed. The new task therefore passes the predeclared causal memory
necessity assay. Physical compression also causes material sequence KL in all
five seeds. These address the previous concern about negligible suffix risk
while the relevant original memory was protected.

## Why the result still fails

Seeds 11,37,51 have teacher-forced answer accuracy 76.18%,86.28%,83.76%, below
90%; their P-generation answer accuracy is 75.59%,85.94%,83.69%.
Local attention error ratios are 1.257,1.362,1.771, above 1.25. Seeds 23 and
71 pass both conditions, with P-generation accuracy 99.90% and 100% and local
ratios 1.006 and 1.005. All seed-level failures remain in the primary aggregate.

| Prerequisite-qualified seed | Policy | KL(Q||P) | Full Fisher | Full relative error |
|---|---|---:|---:|---:|
| 23 | Block | 54.0675 | 0.005328 | 99.990% |
| 23 | Alternating | 51.7728 | 0.004751 | 99.991% |
| 71 | Block | 77.3468 | 0.496232 | 99.358% |
| 71 | Alternating | 78.4717 | 0.720007 | 99.082% |

Thus training failures alone cannot explain away the failed approximation.
Including cross-event terms does not restore finite-endpoint fidelity in these
two qualified realizations. The recorded first-order JVP and Gram arithmetic
are internally consistent, but that is not a bound on nonlinear remainder.
The data show breakdown of this local quadratic approximation at the physical
endpoint; they do not identify a unique causal source of the remainder.

All-five KL is 49.8354/50.1825 for Block/Alternating. Block is larger in only
2/5 seeds; both qualified seeds give different ordering directions. The new
task does not reproduce Round 13's large and consistent temporal separation.
Its scope, freshly independent writes and response geometry differ from the
histogram task. This does not invalidate that earlier finite result or prove
that temporal covariance is irrelevant in every trained model.

## Local-amplitude diagnostic remains secondary

At theta=.25, qualified seed 23 has KL 0.000557/0.000466 versus scaled Full
0.000319/0.000297; qualified seed 71 has KL 0.362992/0.501249 versus Full
0.008976/0.008948. Their sampled answer accuracy remains at least 99.6%, yet
neither meets the 25% approximation threshold for both policies. These are
separate Q-law amplitude diagnostics, not physical compression results:
the smooth cache has 200 records versus 184 exact records. They cannot rescue
the theta=1 endpoint or be reported as a memory-saving regime.

## One-shot control and the Round 7 boundary

The one-shot control compresses an exact prefix at the eighth WRITE's vb,
before its delayed READ, n=90 to m=45, once independently per layer. Subsequent
KV belongs to each model's own stream and is never compressed again. Both
KL directions use the appropriate sampled law and a shadow on the same tokens.
The full identity has zero KL in both directions for every cell.

GS/uniform KL(Q||P) is 11.7477/15.7690, and KL(P||Q) is 6.83986/6.48423.
GS wins per seed in 3/5 and 2/5 respectively; individual prompt/summary cells
have large dispersion. The lower Q-direction mean is not a consistent
advantage across seeds or directions, and KL remains substantial. No theorem
constant or small-risk condition was certified, so this does not refute the
conditional Round 7 law-transfer result. The one-shot arm saves 45 total
records and the continuous arm saves 16; comparisons between their absolute
risks cannot isolate compression timing at a matched total budget.

## Evidence and closure

The [frozen report](evaluation/REPORT.md) and [summary](evaluation/summary.json)
are the minimal aggregate entry points. All 320 continuous, 720 one-shot,
10 P-quality and five ablation rows, plus 270 NPZs, are retained.
[Record-only verification](release_verification.json) passes frozen source
hashes, coverage/uniqueness, saved KL/Fisher sums, full/diagonal Gram identities,
and cache masses/cardinalities; it does not replay checkpoints or prove the
scientific interpretation. NPZs and JSONL rows are secondary evidence.

The [terminal closure](CLOSURE.md) is final for this experimental sequence.
The defensible retained contribution is the narrow Round 13 histogram
mechanism diagnostic plus explicit negative transfer boundaries and the
separately conditional Round 7 theory. Broader trained-model quantitative
transfer, useful certificates, or a stable new compression advantage have not
been established. No claim of publication readiness follows from this run.
