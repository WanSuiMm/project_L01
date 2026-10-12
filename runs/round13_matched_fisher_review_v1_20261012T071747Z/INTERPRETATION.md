# Round 13: cross-event terms explain the finite forward-KL contrast

This is a post-completion interpretation of
`round13_matched_fisher_v1_20261012T062406Z`, preserving its generated artifacts.
The run completed on 2026-10-12 at 06:44:26 UTC in 634.65 scientific seconds,
with 640/640 metric rows, 40 trace files and 80 event Grams.

## Result and what strengthened

At T=129 and amplitude 1, actual selected-history KL(Q1||P) is 1.216538 for
Balanced Block and 0.0997674 for Alternating. Full cross-event Fisher gives
1.220090 and 0.0997077; diagonal-only gives 0.178589 and 0.181480. Their
Block/Alternating ratios are 12.1937, 12.2367 and 0.984066. Both full-accuracy
and full-versus-diagonal screens pass in all five models for each policy.

| Policy | Full relative error across seeds | Diagonal relative error across seeds |
|---|---:|---:|
| Balanced Block | 0.0821% to 0.5639% | 84.98% to 85.62% |
| Alternating | 0.1085% to 0.2522% | 67.47% to 92.27% |

These relative errors compare the phase/prefix mean prediction with the
phase/prefix mean actual risk within each seed. They are not averages of
cell-wise percentage errors. Training seeds are the independent units.

Round 11's Fisher probe used T=32 and KL(P||Q), separately from the T=128
KL(Q||P) primary result. Round 13 now aligns the horizon, direction and
histories with the finite primary risk. It differentiates every event
through both layers and subsequent suffix KV updates, without a fitted scale.

The Gram entry sum equals the full quadratic; its diagonal trace equals the
diagonal-only quadratic. Net off-diagonal energy is +1.04150 for Block and
-0.0817727 for Alternating. Interactions therefore amplify the Block result
and cancel part of the Alternating result in this decomposition. Similar
event-wise diagonal energy alone misses the approximately twelvefold contrast.

## Relation to Round 12

Round 12's real prefix token-bias energy ratio was 11.45 versus actual KL
ratio 11.17, despite only a 1.04 terminal token-bias ratio. That evidence
showed why the entire trajectory matters. Round 13 supplies a more direct
output-sensitive quadratic explanation on the selected histories. It does
not show that a scalar token histogram is the sole mediator.

The selected subset is fixed at the first 16 of 64 saved paths per phase,
selected before the run. Its actual means are +1.22% (Block) and -7.31%
(Alternating) from the original all-path means 1.201930 and 0.107638. The
original 11.17 ratio and selected 12.19 ratio are both reported. Small
same-history Fisher errors do not remove Monte Carlo uncertainty. No new
confidence interval or generalization claim is made.

## Exact interpretation and limits

The Fisher metric is the conditional P-logit quadratic at theta=0, averaged
over saved Q1 histories. It is not the sequence Fisher expectation under P
or the sequence-KL Taylor coefficient at zero. Endpoint conditional KL chain
sums on these Q1 paths estimate KL(Q1||P); they are not an exhaustive tree or
KL of a mixture over hidden schedules. Full Fisher and actual risk share
histories, so the result is an offline explanation rather than heldout forecast.

Amplitude 0.25 gives replay KL 0.0761868/0.00623222 and full predictions
0.0762557/0.00623173. Those are conditional diagnostics on the same Q1 inputs,
not full-sequence KL under Q0.25. Predictions include amplitude squared once.

No unique causal mechanism, useful certificate, general Transformer theorem,
new compressor, novelty verdict or systems benefit follows. The old prefix
and START stay exact; only generated suffix pairs are halved. Round 9's
half-cache certificate failures remain unchanged. This finite histogram task
is now a stronger mechanism demonstration, while transfer beyond its training
distribution and models remains untested. Further experiments require a new
protocol and are not part of this publication check.

## Evidence route

1. [Canonical results](../../RESULTS.md), then [generated report](../round13_matched_fisher_v1_20261012T062406Z/REPORT.md).
2. [Per-seed summary](../round13_matched_fisher_v1_20261012T062406Z/summary.json) and [frozen protocol](../../experiments/PROTOCOL_ROUND13_MATCHED_FISHER_V1.md).
3. [Recorded-evidence verification](verification.json): source/input/checkpoint bindings, coverage, trace arithmetic and seed screens. It does not replay model weights or prove derivatives/theorems.
4. [Endpoint figure](../round13_matched_fisher_v1_20261012T062406Z/analysis/round13_endpoint.png) and [prefix figure](../round13_matched_fisher_v1_20261012T062406Z/analysis/round13_prefix.png).

Raw `metrics.csv`, `traces/*.npz` and input manifests are secondary evidence;
begin with the aggregate. Operational receipts and model weights stay outside
the public release.
