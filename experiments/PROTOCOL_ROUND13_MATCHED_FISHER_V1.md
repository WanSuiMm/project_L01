# Round 13: matched-horizon forward-KL and event cross terms

Frozen before scientific launch, 2026-10-12. This finite output-geometry
diagnostic follows Round 12. It uses original trained checkpoints and saved
histories, with no training, new token generation, fitted coefficient or
calibration against scientific errors. Earlier certificate failures stand.

## Question and fixed inputs

Can exact full-unroll, output-sensitive first-order geometry track the
Balanced Block versus Alternating full-sequence KL gap at T=129, in the
same Q-to-P direction and on the same generated histories as actual risk?
Does dropping cross-event terms lose this explanation?

Use the five original Round 6 two-layer, width-64 histogram models,
checkpoint hashes and qualification metadata from the existing manifest.
Use all four n=64 prefixes from
`round12_balanced_order_v1_20261012T052457Z`. For each policy and antipodal
phase, select path indices 0,...,15 within the saved phase. Saved NPZ rows
are phase-plus indices 0,...,63 then phase-minus indices 64,...,127.
Selection preserves the original common-uniform pairing across policies.
Report selected-path actual risk separately from the original 64-path mean.

Before the formal run, a CPU timing calibration may reduce paths per phase
from 16 to 8 **only if** 40 times one full-length condition's probe runtime
exceeds 2,160 seconds. This rule depends only on runtime, not scientific
values. Record the calibration and freeze the effective configuration before
formal launch. Keep five models, four prefixes and T=129 in either case.
Exact event derivatives are computed in chunks of eight using batched JVPs;
there is no stochastic trace estimator or event subsampling.

## Smooth interpolation and quantities

At event e, replace its physical pair by three atoms with masses
(1-theta_e, 1-theta_e, 2 theta_e), choosing the original endpoint by the
same token/sign rule as Round 12. At theta=0 this is exact P; at theta=1 it
equals physical Q1 pair thinning. The old prefix and START stay exact.
Unroll all layers and suffix KVs on each fixed discrete Q1 history.

Let z_t(theta; y) be logits on fixed history y, and
J_e,t(y)=partial z_t/partial theta_e at the all-zero strength vector.
Signs are already encoded in the endpoint selectors of J_e; do not
multiply the actual signs into these derivatives a second time. Define

    F_full(y) = 1/2 sum_t Var_{P_t(.|y_<t)}[sum_e J_e,t(y)]
    F_diag(y) = 1/2 sum_t sum_e Var_{P_t(.|y_<t)}[J_e,t(y)].

Compute every one of the 64 event tangents exactly through the full unroll.
Check their sum against an independent all-ones strength JVP. Save the
per-phase 64x64 half-Fisher Gram; its entry sum equals mean F_full and
its diagonal trace equals mean F_diag. This directly records cross terms.
Save pathwise prefix cumulative quadratic energies and replay KL values.

At amplitude a, replay z_t(a,...,a) on the same saved inputs and evaluate
**KL(Q_a,t || P_t)** for the full three-token categorical distributions.
At a=1, averaging the conditional sum over these Q1 histories estimates
the full-sequence KL(Q1 || P) for the selected ensemble. At a=.25, the
same operation is only a Q1-history-averaged conditional replay diagnostic;
it is **not** full-sequence KL(Q_.25 || P), whose histories would follow
Q_.25. Quadratic predictions scale by a^2 exactly once.

The Fisher metric at theta=0 is averaged over **Q1 histories**, not P
histories. Name it a conditional Fisher quadratic averaged over Q1
histories. It is neither sequence Fisher under P nor the sequence-KL
Taylor coefficient at theta=0. Its comparison at a=1 is an endpoint-law
diagnostic, not a heldout forecast or causal mediation proof.

## Primary scoring and screens

At horizons 17,33,65,129 and amplitudes .25,1, average equally over paths
within phase, then phases and prefixes within each training seed.
Training seeds (n=5) are the independent scientific units.

At a=1,T=129 compute, for each seed and policy,

    relative_error_full = |mean(F_full)-mean(actual_KL)| / mean(actual_KL)
    relative_error_diag = |mean(F_diag)-mean(actual_KL)| / mean(actual_KL).

Use differences of means, not averages of per-cell relative errors.
Screen full relative error <=0.25 in at least 4/5 seeds for each policy.
Separately report whether full error is smaller than diagonal error in at
least 4/5 seeds for each policy. Report actual, full and diagonal Block/Alt
risk ratios without a new effect-ratio cutoff. Retain negative outcomes.
Report the .25 diagnostic separately, and centered-logit first-order vector
remainders to distinguish logit linearization error from categorical geometry.
No threshold or coefficient is tuned after seeing the probe result.

The original 64-path endpoint risk accompanies the deterministic subset
estimate. Pathwise traces permit separate Monte Carlo uncertainty analysis;
paths and phases are not extra independently trained models. A passing
screen is a finite conditional diagnostic, not a universal risk theorem,
useful certificate, unique mechanism or practical speedup.

## Verification, coverage and execution

Targeted random-fixture checks cover batched versus individual JVPs,
finite differences, physical/smooth endpoints, and positive/negative
cross-term examples. Original seed-11 smoke uses saved histories truncated
to T=9, two paths/phase and two-direction chunks. It is software validation.
The separate full-length timing calibration is also not a scientific result.

For each formal condition check theta=0 against original full-cache P replay,
theta=1 against physical Q replay, and every conditional KL against its saved
Round 12 path. Maximum absolute parity tolerance is 1e-8. Check Gram and JVP
contractions at the same tolerance. Prefix milestones represent inputs
START,Y1,...,Y_(t-1); the sampled Y129 remains outside the final cache.

Formal coverage: 640 metric rows (5 models x 4 prefixes x 2 policies x
2 phases x 4 horizons x 2 amplitudes), 40 trace files and 80 event Grams.
CPU float64, two Torch threads, 45-minute limit. Save source/config/input
hashes, path indices, model manifests and timings in a new run directory.
Operational host/PID/command receipts stay private. Verified dispatch
completes the launch request; no ongoing monitor is created.

From standalone repository root, always use a fresh output directory:

```text
python -B -m unittest discover -s experiments -p test_round13_fisher.py -v
python -B experiments/round13_matched_fisher.py --config experiments/configs/round13_matched_fisher_v1.json --checkpoint-root PATH_TO_ORIGINAL_MODELS --input-run runs/round12_balanced_order_v1_20261012T052457Z --run-dir runs/NEW_ROUND13_RUN
```

Read summary.json and REPORT.md first, then metrics.csv and input_manifest.json.
Trace NPZ files and Gram matrices are secondary evidence. A zero process
exit code alone does not establish a positive scientific conclusion.
