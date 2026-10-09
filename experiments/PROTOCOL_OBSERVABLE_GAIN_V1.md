# Experiment 0: causal observable-gain pilot, v1

Frozen before formal dispatch, 2026-10-10. No training or weight updates.

## Question and competing explanation

Do four already available query positions add useful information about future
distribution drift beyond a current-position output-KL proxy? A current Fisher
quadratic matching current KL is a mathematical sanity check, not the endpoint.
No future token, future Jacobian, or heldout outcome is used to score an edit.

The direct-residual proxy is inspired by Behavior-Preserving KV Cache Compression
(arXiv:2610.06479), not a replication of its complete selection algorithm. The
stronger exact current KL is an additional reference. This pilot cannot establish
superiority over that paper's full method or independent novelty.

## Inputs and independent units

One frozen Qwen2.5-7B-Instruct snapshot, BF16, evaluation mode. Use 24 distinct
WikiText-2 raw validation articles, deterministically sampled with seed 20261010.
The first 8 sampled articles are calibration; the remaining 16 are heldout.
Each contributes one prefix, at most 384 model tokens including the instruction.
The input preparer records the dataset revision, source and prompt hashes,
article IDs, order and split. No answer labels or future continuation text are
inputs. Statistical units are articles/prefixes, not tokens or mask cells.

## Intervention and controls

Select early/middle/late layers by fractions [0.2, 0.5, 0.8]. At one layer at a
time, remove 20% of eligible prefix positions, protecting the first 4 and last
16 positions. Compare a seeded random mask and a mask removing positions with
lowest current mean-head attention. Both have the same eligible-entry budget.
Keep each mask fixed through the continuation; allow later states and appended
KV to evolve under that intervention.

Eviction is implemented by hard masking attention to the selected entries.
For retained positional embeddings this has the attention behavior of deletion,
but retains tensor storage. Therefore no memory reduction, throughput, or
end-to-end compression speedup is measured. The first four recent-query replays
start from a shared earlier cache: prior states before this intervention point
are unchanged. This is one cache intervention, not repeated streaming compression.

The small smoke checks no-mask replay identity and the chosen-layer attention
renormalization against a direct deletion calculation. It is separate from
scientific results and never contributes to calibration or heldout statistics.

## Causal scores

Using full-cache pre-eviction statistics at the last four observed prefix
positions, compute the deletion-induced attention-output difference. Project
it through the attention output matrix and the first derivative of the final
RMSNorm and LM head, retaining only the direct residual path. Compute proxy
KL in direction compressed || full. This is a causal approximation; downstream
attention/MLP responses are omitted and are a potential source of failure.

Baseline: a fixed ridge regression from log10 current proxy KL to log10 future
KL. Candidate: the same regression, additionally using log10 mean and maximum
of the four recent proxy KL values. Exact current KL provides a separately
calibrated reference. Clamp KL at 1e-10 for logarithms; feature standardization
uses calibration data only; ridge coefficient is 1.0, intercept unpenalized.

## Endpoint and sampling

For each of 24 x 3 x 2 = 144 cases, sample one 32-token continuation under the
intervened model, temperature 1, full softmax, no top-k/top-p, fixed horizon
without EOS stopping. Replay the same sampled tokens with the complete model's
own unmodified cache. At each shared prefix compute KL(Q_t || P_t).

Primary outcome is the sum over the 32 *future* prediction positions; current
position t=0 is excluded. Record the whole short KL trajectory and sampled token
IDs. This is a Monte Carlo conditional-drift endpoint, not a measured sequence
TV or a uniform generation certificate.

Primary comparison: heldout squared prediction error in log10 future KL,
candidate versus current-proxy baseline, first averaged within each prefix.
Report Spearman correlation and comparison with exact-current-KL separately.
Use 2000 paired prefix bootstrap samples with seed 20261010 for a 95% interval.

`PILOT_SUPPORT` requires at least 10% mean prediction-MSE reduction and a
bootstrap lower confidence bound above zero. Otherwise use
`NO_HELDOUT_SUPPORT`; insufficient data or runtime failures remain incomplete.
This threshold is a pilot triage rule, not a confirmatory generalization claim.

## Cost and interpretation

Record model/source/config/input provenance, GPU peak allocation and wall time.
The formal job is capped at 90 minutes. Proxy computation time is diagnostic;
this harness does not implement an optimized online scoring system.

Failure to improve heldout prediction stops this recent-query proxy candidate;
it does not disprove all output-relevant compression. Future reactivation alone
is not failure. Oracle-only success would not establish an online method.
No full Jacobian/Fisher matrix or BaseCompress implementation is required.

References: https://arxiv.org/pdf/2610.06479 ;
https://arxiv.org/html/2607.01520v1 ;
https://huggingface.co/datasets/Salesforce/wikitext
