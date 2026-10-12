# Incremental review: Round 12 trajectories and Round 13 cross-event geometry

Review base: `dcf36d77e1e2b8be091b67ee27c3d893f140b0ba` (published Round 11).
Evidence head: `b66b5ea0fb68f1537fd75031ee76b64e8f70e4ac`.
This later commit changes only this handoff; the evidence head stays fixed.

## Minimal reading order

1. [RESULTS.md](RESULTS.md), Round 13 then Round 12: canonical means, screens and limits.
2. [Round 13 interpretation](runs/round13_matched_fisher_review_v1_20261012T071747Z/INTERPRETATION.md), then its [generated report](runs/round13_matched_fisher_v1_20261012T062406Z/REPORT.md) and [summary](runs/round13_matched_fisher_v1_20261012T062406Z/summary.json).
3. [Round 13 protocol](experiments/PROTOCOL_ROUND13_MATCHED_FISHER_V1.md) and [GPT_CONTEXT.md](GPT_CONTEXT.md): exact interpolation, history law, quantities and source symbols.
4. [Recorded-evidence verification](runs/round13_matched_fisher_review_v1_20261012T071747Z/verification.json); [Round 12 interpretation](runs/round12_balanced_order_review_v1_20261012T060523Z/INTERPRETATION.md) and [joint record check](runs/round12_balanced_order_review_v1_20261012T060523Z/joint_bias_kl_review.json).

Raw NPZ, event CSV and the large Round 12 summary are secondary evidence.
Do not open those first. Round 12's frozen interpretation predates Round 13;
its proposed Fisher follow-up is now completed.

## Changed evidence and claims

Round 12 compares 32 positive then 32 negative suffix-thinning events against
alternating events, both antipodal phases, T=129, four prefixes, and the five
original two-layer histogram models. Each phase has 64 sampled Q paths.
Actual KL(Q||P) is 1.201930 versus 0.107638 (11.17x). Terminal token-bias
squared is only 1.04x, while integrated prefix token-bias squared is 11.45x.
The actual trajectory matters; terminal bias misses the finite risk contrast.
Local mean-error comparability and KL ordering pass all five models.

Round 13 retains T=129 and KL(Q1||P), using the deterministic first 16 saved
paths per phase without training or new token generation. Every event JVP
propagates through the full causal unroll and both layers. No fitted scale is
used. Actual selected-history risk is 1.216538/0.0997674; full cross-event
Fisher is 1.220090/0.0997077; diagonal-only is 0.178589/0.181480.
The Block/Alternating ratios are 12.1937, 12.2367 and 0.984066.

Full mean per-seed relative error is 0.322%/0.171%, versus 85.32%/82.35%
diagonal-only. Both predeclared screens pass 5/5 seeds for each policy.
Net off-diagonal contributions are +1.04150 (Block) and -0.0817727 (Alternating).
This strengthens the finite same-history output-geometry explanation of the
policy contrast, closing Round 11's short-horizon/opposite-direction gap.

The quadratic is conditional P-logit Fisher averaged over Q1 histories,
not sequence Fisher under P or the sequence-KL Taylor coefficient at zero.
Actual KL and geometry share histories; this is not heldout forecasting or
unique causal mediation. Amplitude 0.25 is Q1-history replay only, not
KL(Q0.25||P). Predictions include amplitude squared exactly once.
The selected actual means differ by +1.22%/-7.31% from the all-path means;
retain that Monte Carlo sensitivity beside the tiny same-history errors.
Five training seeds are independent; paths, phases and prompts are not.

## Validation and unchanged claims

All 640 metric rows, 40 traces and 80 Grams completed. Two Round 13 and four
Round 12 targeted tests pass. The recorded-evidence verifier binds source,
config, checkpoint metadata and all selected input hashes; checks trace/CSV,
Gram sum/trace and KL-chain arithmetic; and recomputes seed means/screens.
It does not replay model weights, verify JVP derivatives or prove a theorem.
The earlier original-model smoke checks provide separate implementation evidence.

Round 11, Round 9 certificate failures and directional follow-ups, Round 7,
Round 6 and Experiment 0 remain unchanged. The old prefix and START stay
exact; only eligible generated suffix pairs are halved. No whole-cache
half-budget, useful certificate, universal Transformer law, cache-space
lower bound, novel compressor, novelty verdict or practical speedup follows.

## Reviewer questions

1. Is the conditional Fisher metric and its Q1-history averaging distinguished
   correctly from P-law sequence Fisher and Q0.25 sequence risk?
2. Does the full-versus-diagonal comparison support the finite interaction
   explanation without implying a unique causal mechanism or heldout predictor?
3. Are deterministic subset sensitivity and the five-model independent unit
   reported adequately beside the sub-percent same-history approximation errors?
4. What minimal transfer beyond this histogram setting would materially
   strengthen the scientific contribution? No such experiment has run here.
