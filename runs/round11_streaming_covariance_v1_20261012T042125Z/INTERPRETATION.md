# Round 11 interpretation

Both frozen descriptive screens pass in all five original two-layer
histogram models. At T=128 the estimated mean fixed-schedule KL(Q||P) is
2.730185 for coherent, 0.175011 for IID and 0.100431 for alternating
thinning. Their per-model mean direct local attention squared errors differ
by at most 1.18%. Thus identical physical record budgets and similar mean
local distortion can coexist with very different generated sequence risk.

The intervention changes the time ordering of positive/negative thinning
decisions. All policies balance the unconditional sign marginal using
antipodal schedules. Their conditional histories and vector error laws
need not agree. The local mean screen is evidence against a simple
mean-distortion explanation, not proof that all other factors are matched.

The full cross-event Fisher quadratic is very accurate for the distinct
fixed-P-history experiment at T=32. At amplitude 0.25, average relative
error is 0.04810% for coherent and 0.01206% for alternating, versus 79.66%
and 78.50% for a diagonal-only event sum. This supports the relevance of
cross-event terms in this local reverse-KL response. Both finite KL and
quadratic are evaluated on the same P histories; there is no heldout
prediction or cheap online proxy.

Primary measurement samples Q histories and sums Q||P conditional KL at
T=128. Fisher measurement samples P histories and sums P||Q conditional KL
at T=32. Both directions share the infinitesimal Fisher metric, but the
experiment does not establish that the short reverse-KL diagnostic
explains the entire long-horizon forward-KL gap. That bridge remains open.

Only policy ordering transfers from the analytical temporal-covariance
construction. No asymptotic growth rate was established. Alternating KL
still grows from 0.00495 at T=16 to 0.10043 at T=128; it is not demonstrated
to be uniformly bounded in horizon.

The old n=64 prefix and START remain exact. Only new suffix pairs are
replaced by one original record of weight 2. At the final T=128 prediction,
each layer has 129 records instead of 192, with equal logical mass. This
is a physical record reduction for this intervention, not a measured
memory-byte saving or inference acceleration. Interpolation amplitudes
below one store extra atoms and are diagnostic rather than equal-budget
compressed caches.

The independent unit is the training seed (five models). Four fixed
prefixes, finite token paths and eight IID schedules restrict scope.
These results support a trained toy mechanism study, not a general NLP
claim, certification result, new compression algorithm or novelty claim.
Round 9's failed half-cache certificate and all earlier evidence remain
unchanged. No retraining or outcome-based tuning occurred.

Read [RESULTS.md](RESULTS.md), [summary.json](summary.json), then
[release_verification.json](release_verification.json). The larger
event_metrics.csv is secondary. The audit checks recorded hashes,
coverage and arithmetic without replaying model weights.
