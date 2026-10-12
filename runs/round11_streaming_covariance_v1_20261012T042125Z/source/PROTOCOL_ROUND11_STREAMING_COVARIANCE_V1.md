# Round 11: temporal compression policies in frozen trained decoders

Frozen question: can temporal correlation between equally budgeted thinning
decisions change autoregressive fidelity in the original trained two-layer
histogram decoders, and does the full reference-path Fisher response explain
more than a sum of isolated event costs?

This is a new mechanism experiment, not a continuation of the Round 9
certificate gate. No model training, parameter search or certification is
performed. Round 6/7/9 evidence is read-only and retains its original verdicts.

## Models, inputs and intervention

Use the five original width-64, two-layer checkpoints (seeds 11,23,37,51,71),
with hashes and quality metadata from `configs/round9_expected_checkpoints.json`.
Generate four fixed length-64 histogram prefixes with the original generator
and the frozen prompt seed. Retain the complete old prefix and START exactly.
Positions retain their learned absolute indices; all predictions remain within
the existing position table. Full three-token softmax includes the rare
continuation marker, with no top-k, top-p or EOS stopping.

The new streaming intervention applies to generated suffix KV, not the old
prefix. At input steps 2,4,6,... (START is input step 0), the newest two
generated records at each layer are replaced by one of those original records
with weight 2. Implementation computes the current query/key/value, appends
the current record, and compresses the pair BEFORE that layer's attention.
Thus the pair for generated tokens y1,y2 first affects prediction y3. The
upper layers and subsequent suffix records use each decoder's own states.
This insertion is causal; it does not reuse future tokens or reference KVs.

The plus endpoint retains the larger token ID of the pair, with an earlier
position tie-break; minus retains the other record. Token IDs are known at
compression time. This is a predeclared label convention, not learned
feature balancing or a proposed general compression algorithm.

All three policies have identical record counts at every prediction:
`n + 1 + ceil(step/2)` per layer. Logical mass is `n + step + 1`.
They halve eligible generated pairs, not the entire cache. Keys differ, so
mass preservation does NOT imply preservation of the softmax denominator.

- Coherent: enumerate the two phases, all-plus and all-minus.
- Alternating: enumerate plus/minus alternating and its antipode. The name
  does not assert cancellation of trained-model vector errors.
- IID: four frozen independent sign strings and their four antipodes.

## Endpoint A: actual generated sequence risk

For each model/prefix/policy, use 128 token paths: 64 per coherent/alternating
phase, or 16 per each of eight IID schedules. Common uniform random numbers
couple policy comparisons; every Q still samples from its own full softmax.
P replays those tokens using its own complete cache. Record conditional
Q-to-P KL sums at T=16,32,64,128. This estimates the average of fixed-schedule
sequence KLs; it is NOT the KL of a mixture over hidden compressor schedules.
The IID result is a finite eight-schedule estimate, not exact expectation.

Measure the direct local attention change before/after replacing the pair on
the SAME current Q query/cache, squared signed key/value moment changes,
relative denominator changes, and cumulative signed feature energy. Retain
per-event means and quantiles as well as schedule-level values. Across Q
policies, local error distributions need not match because histories and
upper-layer KVs change. Matching a mean is only a comparability screen.

The independent statistical unit is training seed. Average equally over
prefixes and compressor phases/schedules inside each model, then report
paired differences and training-seed intervals. Token paths, events and
compression schedules are not additional independent trained models.

Primary descriptive transfer screen: at T=128, coherent > IID > alternating
mean sequence KL in at least four of five models. Separately require, in at
least four models, max/min policy mean direct local attention squared error
<=1.25. An ordering without this local comparability screen is reported as
`POLICY_EFFECT_WITH_LOCAL_DISTORTION_CONFOUND`, not isolated covariance.
Failure of the ordering is `NO_ROUND10_ORDER_TRANSFER`; it does not refute
the analytical Round 10 construction. Ratios with zero denominators stay
explicit; no epsilon is added to manufacture a finite ratio.

## Endpoint B: fixed-reference causal Fisher mechanism

Use the first two frozen prefixes of every model, T=32, and 16 independently
sampled P histories. Hold discrete histories fixed. Introduce one smooth
strength theta_j per compression event: each original pair retains masses
1-theta_j,1-theta_j and a selected endpoint atom has mass 2 theta_j.
Every interpolated decoder computes its own hidden states and suffix KVs.
At theta=0 it is exactly P; at theta=1 it is the physical thinning endpoint.
Intermediate amplitudes are diagnostic mixtures with extra atoms, not the
physical compression budget.

At theta=0, compute all event Jacobian-vector products through the complete
unroll, including upper layers and later KVs. Opposite choices give exactly
opposite signed atom derivatives on each fixed P history. For a sign string
s, compare full Fisher `0.5 E_P sum_t (sum_j s_j J_tj)' F_t (sum_j s_j J_tj)`
with its diagonal-only version `0.5 E_P sum_tj J_tj' F_t J_tj`.
The ideal IID expectation equals the diagonal expression analytically;
coherent and alternating retain their respective cross terms. Report both
the exact schedule-covariance predictions and the sampled IID sign average.

For the fixed schedules, independently evaluate finite reverse sequence KL
on those P histories at amplitudes 0.25 and 1.0. This is an estimate of
P-to-Q KL, not arbitrary teacher-forced Q-to-P KL. Compare it with the
amplitude-squared full and diagonal Fisher predictions. Endpoint B uses
independent P draws from Endpoint A and is an offline diagnostic, not a
cheap future-risk predictor or a reference-only certificate.

Mechanism screen on coherent/alternating at amplitude 0.25: full Fisher
has lower mean relative prediction error than diagonal Fisher and mean
relative error <=0.25 in at least four training seeds. The relative error
uses a fixed 1e-12 floor, reported zero-scale cases remain identified.
Failure is retained as `NO_FINITE_FISHER_EXPLANATION`.

## Checks, provenance and outputs

Before formal launch, run only the targeted streaming-core tests and one
small smoke with the original seed-11 checkpoint. Check positive masses,
physical record counts, theta=0 original-model identity, theta=1 physical
endpoint identity, and finite-difference/JVP agreement. No outcome-based
model/prefix selection or retuning follows the smoke.

Freeze source/configuration hashes, checkpoint hashes, prefixes and seeds in
a new `runs/round11_streaming_covariance_v1_<UTC>/` directory. Write compact
schedule metrics, per-event aggregates, Fisher rows and a generated summary.
Keep operational host/PID/paths/launch receipts under `_private`; do not
publish them. An automatic completion analysis is part of this job; no
ongoing monitor is requested. The time limit is 45 minutes.

No general trained-Transformer theorem, novel compressor, useful certificate,
asymptotic trained-model rate, inference speedup or NAACL acceptance follows
from either endpoint. Stable endpoint ordering and a valid linear mechanism
are distinct findings; all negative and confounded outcomes are retained.
