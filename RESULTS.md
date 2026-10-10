# L01 results: Round 6 completed toy study

**Execution: COMPLETE. Scientific interpretation: analytic consistency and
qualified histogram evidence; recall training is unreliable and the global
certificate is uninformative.** This is not a general trained-Transformer
validation or evidence of publication readiness.

The formal run is `round6_toy_v1_20261010T061030Z`. The completed endpoint is
the [20261010T073818Z snapshot](runs/round6_toy_v1_20261010T061030Z/snapshots/20261010T073818Z/scientific_summary.json),
not the earlier launch-time analysis in the run's top-level `analysis/`.
Experiment 2 completed in 3,779.20 seconds (about 63 minutes); final analysis
finished on 2026-10-10 at 07:16:53 UTC.

## Canonical Round 6 endpoints

| Component | Completed evidence | Interpretation |
|---|---|---|
| Exact analytic construction | 160 configurations; 40 short-horizon sequence enumerations; all 12 recorded gates pass | Supports the implemented epsilon-squared and nT/(n+T) scales |
| Trained evaluation | 20/20 models; 22,080/22,080 prompt rows; 276/276 complete groups | All five training seeds per task/depth retained; no missing groups |
| Histogram training | 10/10 pass; Bayes NLL gap 0.000181–0.000473 versus threshold 0.03 | Competent models for this simple synthetic task |
| Associative recall training | 3/10 pass: 1/5 at depth 2, 2/5 at depth 4; threshold answer top-1 0.90 | Seven failures prevent a robust learned-recall interpretation |
| Global sufficient bound | No tighter than the model-specific universal bounded-logit cap in 840/840 applicable group-seed cases | No useful uniform-generation certificate; Pinsker reaches 1 in all 840 cases |

The primary statistical unit is the **training seed**, with five seeds per
task/depth. KL is the sum of conditional Q-to-P KL on Q-sampled histories,
averaged over eight paths per prefix, 16 prefixes per seed, then seeds. It
estimates sequence KL for the fixed empirical prompt set. Error bars are
two-sided 95% Student-t intervals across the five seeds. The rows and paths
are not independent training replicates. Failed-quality seeds are included
in every primary aggregate; no outcome-dependent retuning was performed.

## What the curves show

For the exact construction, `KL / (epsilon^2 * n*T/(n+T))` ranges from
**0.110134 to 0.124873** across all 160 configurations. The finite sums display
short-horizon growth and long-horizon saturation. This validates the named
construction and its arithmetic, not typical trained-model behavior or a
minimax cache-space lower bound.

The trained histogram results show slowing accumulation over the measured
horizons. Representative five-seed mean sequence KL below uses prefix n=64,
summary m=16, rho=1; both summaries preserve mass.

| Histogram depth / method | T=8 | T=32 | T=128 |
|---|---:|---:|---:|
| 2 layers / uniform weighted subset | 0.05956 | 0.18347 | 0.39229 |
| 4 layers / uniform weighted subset | 0.06108 | 0.19372 | 0.42458 |
| 2 layers / positive cluster | 0.00001274 | 0.00003532 | 0.00006847 |
| 4 layers / positive cluster | 0.00000500 | 0.00001494 | 0.00003239 |

For n=64, the coordinate nT/(n+T) is 7.11, 21.33 and 42.67 at these horizons.
These finite curves are consistent with slowing accumulation; T<=128 does
not establish the eventual plateau. The very small cluster-summary errors
are descriptive evidence for this task and compressor, not a general bound.

The rho-mixture precision diagnostic is also informative on histogram. Across
depths and horizons, mean KL at rho=0.25 is **0.0595–0.0618** times its rho=1
value, and at rho=0.5 is **0.2412–0.2480** times it, close to rho squared.
These mixtures store **n+m=80** prefix records per layer rather than m=16;
rho is intervention amplitude, not a certified all-query epsilon. Recall
ratios are less consistent and most recall models failed the training gate.

The global audit records log C_model from **139.22 to 1869.95** and log coarse
uniform epsilon from **12.70 to 261.61**. The theorem may be valid while these
global sufficient constants have no useful explanatory force in this setting.
Observed-query numerator/denominator errors remain sampled diagnostics;
substituting them into the theorem would not certify its assumptions. No
constant was fitted to observed KL.

## Evidence and reproduction route

1. Read the [completed derived report](runs/round6_toy_v1_20261010T061030Z/snapshots/20261010T073818Z/analysis/RESULTS.md)
   and [three figures](runs/round6_toy_v1_20261010T061030Z/snapshots/20261010T073818Z/analysis/multilayer_autoregressive_fidelity.png).
   Figure 3 retains both 25% and 50% budget curves for each method and labels
   failed task-quality coverage. The figure's t intervals can extend below zero;
   KL estimates themselves are nonnegative up to numerical tolerance.
2. Use [scientific_summary.json](runs/round6_toy_v1_20261010T061030Z/snapshots/20261010T073818Z/scientific_summary.json)
   for all model quality records and planned/observed coverage;
   [analysis/aggregates.json](runs/round6_toy_v1_20261010T061030Z/snapshots/20261010T073818Z/analysis/aggregates.json)
   contains seed-level curves, uncertainty and bound status. Do not open the
   large aggregates or compressed per-prompt endpoints first.
3. [Snapshot provenance](runs/round6_toy_v1_20261010T061030Z/snapshots/20261010T073818Z/provenance.json)
   binds frozen source/configuration and allowlisted endpoint files. The
   [frozen protocol](experiments/PROTOCOL_ROUND6_TOY_V1.md) defines tasks,
   compression, gates and sampling. [README.md](README.md) gives reproduction.

Offline verification `python -B scripts/verify_round6_release.py` passes:
12 frozen source hashes, three endpoint-file hashes, eight derived-analysis
hashes, complete coverage and group aggregation, 2,880 full-cache identity
rows, and the recomputed 160-row analytic grid. The verifier checks recorded
norm-audit fields but does not recalculate norms from private checkpoints.

Actual Q cache tensor bytes include suffix K/V and explicit weights, but exclude
allocator residency and Python overhead. Dense replay/oracle storage is not Q's
compression cost. No practical speedup, end-to-end memory saving, fixed-memory
streaming or full-sequence TV endpoint is established. The suffix grows with T.
The unweighted subset is an out-of-theorem mass-changing control.

The next empirical gap is a reliably learned recall baseline under a new frozen
protocol, preserving these failures. A useful model-dependent propagation
certificate is a separate theoretical gap. Neither is resolved by the current
run. Historical Experiment 0 remains unchanged below.

---

# Experiment 0 result

**Execution: COMPLETE, 144/144 cases. Scientific endpoint:
NO_HELDOUT_SUPPORT.** Adding recent-position information did not improve the
prespecified heldout future-KL prediction objective.

## Primary endpoint

All models below are fixed ridge regressions calibrated on 8 articles. Evaluation
uses 16 different articles, each with 6 interventions. Loss is squared error in
log10 future cumulative conditional KL, averaged within each article and then
across articles. Lower is better.

| Predictor | Heldout prediction MSE | Case-level Spearman |
|---|---:|---:|
| Current direct-residual KL proxy | 0.1040362242 | 0.23936 |
| Current plus recent mean/max proxies | 0.1042276286 | 0.24259 |
| Exact current KL, separately calibrated reference | 0.0949199813 | 0.20129 |

The primary baseline-minus-candidate **loss reduction** is
**−0.0001914044**, or **−0.18398%** improvement. Its paired article-bootstrap
95% interval is **[−0.0004202861, +0.0000445716]** in loss units. The frozen gate
required at least 10% reduction and a positive lower bound; it was not met.
The small observed worsening is not evidence of a statistically established
worsening. The bootstrap resamples articles while conditioning on the one
sampled continuation in each intervention cell; it does not quantify additional
rollout-sampling variance. Spearman is descriptive; no cell-level significance
is claimed.

The `actual_current_oracle` key in the raw summary means **exact current KL**.
It has no future information and is not a future oracle. It is an additional
reference, not the primary gate or a deployable cheap method.

## What ran

| Item | Recorded setting |
|---|---|
| Frozen model | Qwen2.5-7B-Instruct, BF16, eager attention |
| Inputs | 24 distinct WikiText validation articles; 8 calibration / 16 heldout |
| Context | At most 384 tokens including the chat template |
| Intervention | Persistent logical deletion of 20% of eligible entries at one layer |
| Layers | 5, 14, 22, zero-based |
| Masks | Seeded random; lowest current mean-head attention |
| Future horizon | 32 prediction positions, excluding current t=0 |
| Sampling | Temperature 1, full softmax, no top-k/top-p or EOS stopping |
| Completed cells | 48 calibration + 96 heldout = 144 |
| Runtime | 157.13 seconds including worker startup and final analysis |
| Hardware | One NVIDIA GeForce RTX 5090 |

Q samples the continuation with its masked cache. P replays exactly those tokens
with its own unmodified cache; KL direction is Q || P at each shared prefix.
The target sums positions t=1 through t=32: the final distribution is evaluated
after the 32nd sampled token. It is not the KL of exactly the sampled 32-token
string, and it is not an empirical sequence-TV measurement.

## Interpretation and limits

This run gives no heldout support to the **four-token chat-suffix,
direct-residual proxy variant**. The runner selects the final four positions
after applying the chat template. The pinned model's
[template](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/a09a35458c702b33eeacc393d103063234e8bc28/tokenizer_config.json)
ends the user turn and appends an assistant header. These are boundary/header
query positions, not the last four article-content queries. Sparse large proxy
values also dominate the raw recent-window mean; regression uses log-transformed
features. The proxy omits downstream attention and MLP responses. These limit
which mechanism has been tested.

The result does not disprove future observability, establish that current KL is
universally sufficient, or measure the spectrum of a causal observability
Gramian. There is one model, one prompt family, one eviction fraction and one
sampled continuation per intervention. No stable low-sensitivity subspace or
uniform stability certificate has been established.

Masking retains tensor storage. GPU allocation includes model weights and the
diagnostic harness; it is not evidence of cache compression. The scoring method
is inspired by a direct-residual approximation, not a replication or comparison
against the complete Behavior-Preserving compression algorithm. No quality,
memory or speed superiority over that method is claimed.

The decision is to stop this proxy candidate under the frozen protocol. A
different proxy requires a new rationale, protocol and run, preserving this
negative result.

## Evidence

Canonical run: `runs/observable_gain_v1_20261009T190129Z`.

- [Analysis aggregate](runs/observable_gain_v1_20261009T190129Z/observable_gain_summary.json)
- [Execution and hash summary](runs/observable_gain_v1_20261009T190129Z/execution_summary.json)
- [Runtime aggregates](runs/observable_gain_v1_20261009T190129Z/summary.json)
- [Frozen generated report](runs/observable_gain_v1_20261009T190129Z/RESULTS.md)
- [Secondary per-case evidence](runs/observable_gain_v1_20261009T190129Z/cases.jsonl)

Offline replay, trajectory checks and input/source hashes are checked by
`scripts/verify_results.py`. Two focused analysis tests check calibration-only
fitting, split isolation and recovery of a deterministic feature advantage.
The model revision comes from the frozen configuration and launch record;
independent byte hashes of the model weights and tokenizer were not recorded.
The separate GPU smoke checked cache replay identity and attention deletion
semantics; it contributed no scientific data.
