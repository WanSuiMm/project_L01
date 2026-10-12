# Incremental review: Round 11 temporal-policy mechanism

Review base: `ef4cfbcb3aac3293f76f777a012df917077ffe32`.
Evidence head: `045c8b887bf5d4ac030cb14aa480aea619fbc829`.
This later commit changes only this handoff; the evidence head stays fixed.

## Minimal reading order

1. [RESULTS.md](RESULTS.md), Round 11 section: primary means, both screens and scope.
2. [Run interpretation](runs/round11_streaming_covariance_v1_20261012T042125Z/INTERPRETATION.md), then the
   [plot correction](runs/round11_streaming_covariance_v1_20261012T042125Z/plot_correction/README.md).
3. [Frozen protocol](experiments/PROTOCOL_ROUND11_STREAMING_COVARIANCE_V1.md),
   then [GPT_CONTEXT.md](GPT_CONTEXT.md) for source symbols and estimands.
4. [Generated per-seed report](runs/round11_streaming_covariance_v1_20261012T042125Z/RESULTS.md),
   [summary](runs/round11_streaming_covariance_v1_20261012T042125Z/summary.json) and
   [audit](runs/round11_streaming_covariance_v1_20261012T042125Z/release_verification.json).
   The larger per-event CSV is secondary; do not open it first.

## Changed evidence and claims

Five original width-64, two-layer histogram checkpoints were reused with no
training. Four fixed n=64 prefixes per model and 128 sampled token paths per
policy compare coherent, IID and alternating positive pair-thinning schedules.
The old prefix and START remain exact; only eligible new suffix pairs are
halved. Each policy has the same physical record count, not a measured system
memory or speed budget. All 960 primary, 60 Fisher and 7,560 event rows complete.

At T=128 mean fixed-schedule sequence KL(Q||P) is 2.730185, 0.175011 and
0.100431 respectively. All five models have this ordering; per-model
max/min mean local attention squared error is 1.0055--1.0118. Both ordering
and the predeclared mean-local-distortion comparability screen pass 5/5.
Schedules balance unconditional sign marginals, not full error laws on
different generated histories. The estimand is E_R KL(Q_R||P), not mixture KL.

On a separate fixed-P-history probe (T=32, two prefixes, 16 paths), full
cross-event Fisher predicts finite reverse KL(P||Q) at amplitude 0.25 with
mean relative errors 0.04810% coherent and 0.01206% alternating, versus
79.66% and 78.50% for diagonal-only event sums. The local screen passes 5/5.
Prediction and finite KL share P histories: this is an offline explanation.
It does not directly explain the entire T=128 Q||P risk gap.

Only the finite policy ordering transfers from the analytical construction.
No linear/logarithmic/bounded asymptotic rates were tested; alternating KL
still increases over the measured horizons. No general trained-Transformer
theorem, new compressor, useful certificate, systems gain or novelty claim follows.

## Plot correction and verification

Publication review found double amplitude scaling in the original Fisher
figure only. The CSV predictions already include amplitude squared. Current
analyzer code is fixed; use [the corrected figure](runs/round11_streaming_covariance_v1_20261012T042125Z/plot_correction/fisher_predictions.png).
Frozen v1 sources, figures and numeric outputs remain intact; a separate
correction manifest binds the new figure to the original CSV and fixed code.

Four targeted core tests and original-model smoke checks pass. The read-only
audit checks source/payload hashes, complete unique row coverage, seed means,
relative-error arithmetic and both screens. It does not replay weights, token
paths, JVPs, proofs or novelty. Original checkpoints are separately held.

```text
python -B scripts/verify_round11_release.py --input-dir runs/round11_streaming_covariance_v1_20261012T042125Z
python -B -m unittest discover -s experiments -p test_round11_streaming_core.py -v
```

## Unchanged evidence

Round 9 half-cache certificate failures, its one-case JVP and failed remainder
attempt, Round 7, Round 6 and Experiment 0 are unchanged. This is a new
mechanism protocol, not a rescue or continuation of the certification gate.

## Reviewer questions

1. Is E_R KL(Q_R||P) estimated correctly on each Q's own histories, with
   full-cache P replay, causal insertion and exact positions/mass?
2. Do event JVPs include upper layers and later self-generated KV, and are
   full/diagonal Fisher comparisons valid on the fixed P histories?
3. Does the mean-local-error screen plus the separate reverse-KL probe
   justify the stated narrow mechanism claim without over-attributing the
   full T=128 forward-KL gap to covariance?
4. What additional controlled bridge would distinguish a transferable
   mechanism contribution from familiar averaging in a histogram task?
