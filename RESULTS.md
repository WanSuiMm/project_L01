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
