# Round 6 toy protocol, v1

Frozen protocol for formal dispatch, 2026-10-10. This file
defines two bounded experiments for the one-time prefix-KV replacement result
in [THEOREM_AND_PROOF.md](../THEOREM_AND_PROOF.md). It is a protocol, not an
experiment result or a claim that trained models satisfy the theorem's
assumptions. The machine-readable parameter mirrors are
[round6_scaling_v1.json](configs/round6_scaling_v1.json) and
[round6_transformer_v1.json](configs/round6_transformer_v1.json).

The theorem document is a user-provided proof draft. Its referenced
`verify_round6.py`, `verification_results.json`, and `NOVELTY_AUDIT.md` are not
present in this checkout; their historical checks are not claimed as reproduced
by this experiment. The new runners and tests are versioned independently.

The independent scientific unit in Experiment 2 is the training seed. Report
all five seeds and their mean and standard deviation; prefixes are secondary
clusters and generated suffix paths are Monte Carlo replicates, not independent
training units. No search over architecture, optimizer, compressor, or
thresholds is permitted. A failed quality gate is retained in the record,
labelled as a failed gate, and excluded only from interpretation as a
successfully learned task; it is not retrained or silently dropped.

## Source boundary and adaptation

The closest cache-compression protocol is Gelberg et al.,
[Training Transformers for KV Cache Compressibility](https://arxiv.org/html/2605.05971v1),
§2 and Appendix C.3. Their intervention replaces only the prefix cache; later
suffix tokens are appended normally. Their FineWeb evaluation uses a 768-token
prefix and 256-token suffix, 128 examples, and scores suffix perplexity, KL, and
top-1 agreement. Their gradient-fitted cache uses the same ground-truth
prefix–suffix example to fit and evaluate the cache, and its KL objective is
dense-teacher || compressed-student (Appendix C.3, equation there). This Round 6
toy is an adaptation: no suffix-derived fit is used, and the primary KL is
compressed-model || full-model on paths sampled from the compressed model.
Gelberg et al.'s KV-CAT training is a separate continued-pretraining method
(Qwen2.5, 40k steps and 5.24B tokens in Appendix C.1), not the small from-scratch
toy training specified below.

For the associative-recall task, Zoology,
[Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/html/2312.04927),
Appendix E.1–E.2 is the closest synthetic-task source. Its MQAR generator puts
random key–value pairs at the beginning and places repeated keys at
power-law-distributed distances; its synthetic attention model has learned
positions, one head, and two layers. Their synthetic runs use 100,000 train
examples and 3,000 evaluation examples, AdamW with weight decay 0.1, 10% linear
warmup, a four-value learning-rate sweep from 1e-4 to 1e-2, and 64 epochs (batch
64 for their small sequence/model settings). The task here is an explicit
adaptation, not a replication: keys and their episode-specific values occur in
the fixed prefix, then query/answer pairs follow CONT, with query keys sampled
uniformly with replacement. Query-to-source distances vary, but are not
power-law sampled.

Li et al., [Transformers as Algorithms: Generalization and Stability in
In-context Learning](https://proceedings.mlr.press/v202/li23l/li23l.pdf),
§7 and Appendix A, is relevant as an in-context-learning training precedent,
not as a cache-compression protocol. Its standard GPT-2 setup is 12 layers, 8
heads, and width 256. Its experiments use random linear tasks and partially
observed linear dynamics; Appendix A reports 6.4M dynamics trajectories,
batch 64, 500k/100k iterations, Adam, and learning rate 1e-4. Those much larger
settings are not imported into this bounded compression experiment.

## Experiment 1: exact analytic scaling

**Question and estimand.** Does the fixed weighted two-atom construction have
the predicted epsilon-squared sequence-KL scale and nT/(n+T) horizon profile?
The estimand is its exact finite sum of conditional KLs for each configured
(n,T,epsilon); it has no sampling unit.

Use the fixed two-atom weighted-subset construction in §4 of
[THEOREM_AND_PROOF.md](../THEOREM_AND_PROOF.md). For each even
n in {8, 32, 128, 512}, the dense prefix contains n/2 values +1 and n/2 values
-1 with identical zero keys; all queries and appended keys are zero. Replace
the prefix by the two retained atoms +1 and -1 with positive weights
n(1+epsilon)/2 and n(1-epsilon)/2. The total prefix mass remains n, and the
compressed prefix numerator is n epsilon. Appended records have unit mass and
zero value. The binary output logits are (a_t, 0), where

    a_t = epsilon*n/(n+t),  P_t = Bernoulli(1/2),
    Q_t = Bernoulli(sigmoid(a_t)).

Evaluate the full grid epsilon in {0.025, 0.05, 0.1, 0.2, 0.4} and
T in {2, 8, 32, 128, 512, 2048, 8192, 32768}. Compute the exact conditional
KL in float64 and sum it over t=1,...,T; use a stable log-sigmoid/log-sum-exp
implementation and verify the closed-form probability calculation to 1e-10.
There is no Monte Carlo and no confidence interval. Report KL against
h(n,T)=nT/(n+T), and the dimensionless ratio KL/(epsilon^2 h). The analytic
checks are the expected epsilon-squared collapse at small epsilon, the
short-horizon behavior h approximately T when T is much smaller than n, and
the approach to the n-scale plateau when T is much larger than n. Do not
require every finite epsilon curve to collapse exactly.

Controls are: (i) the uncompressed full-cache/oracle comparison, whose KL and
TV are exactly zero; (ii) epsilon=0 with equal n/2 weights on the two atoms,
also exactly zero; and (iii) an append check at every t that the common
start-token and suffix records are appended once with unit mass and the
logical total mass is n+t. The append check guards the time index and denominator
in a_t; it does not rerun compression during generation.

Compute exact sequence TV by enumerating all 2^T binary strings only for
T<=8. Report no TV estimate for longer horizons: an MC KL estimate is not a
TV estimate. There is no fitted constant C_model and no pass/fail criterion
requiring this analytic construction to validate an unrelated trained model.

## Experiment 2: trained causal decoders

**Question and estimand.** For these frozen small task models, what is the
sequence KL caused by a one-time prefix-cache intervention when the compressed
model generates its own continuation and the full model replays the same
sampled tokens through its own cache? The primary estimand is the mean
Q-conditional sequence KL over the 16 fixed held-out prefixes at each n and
the compressed-model continuation distribution. The training seed is the
independent scientific unit (five per task and depth); prefixes are secondary
clusters and the eight paths per prefix are Monte Carlo replicates.

### Tasks and sequence construction

Train and evaluate each task as a separate full-softmax next-token model. Each
episode has n data tokens, then one uncompressed CONT token, then a 128-token
suffix. Use n uniformly sampled from {16, 64, 128}; the longest sequence has
257 positions (128 prefix + CONT + 128 suffix). No EOS stopping is used.

* **Binary histogram.** Draw one theta ~ Beta(2,2) per episode. Draw the n
  prefix symbols and 128 suffix symbols independently as Bernoulli(theta).
  The exact Bayes suffix predictor, given the prefix and previous teacher-forced
  suffix symbols, assigns probability (2+c1)/(4+c0+c1) to symbol 1, where c0
  and c1 count zeros and ones observed so far. Use the three-token alphabet
  {0, 1, CONT}; train next-token loss only on the 128 suffix positions.
* **Associative recall adaptation.** Use disjoint key and value alphabets of
  size 64 each, plus CONT. Set D=n/2. In each episode, sample D distinct keys
  without replacement, assign each key a value drawn independently and
  uniformly with replacement, and serialize the D key–value pairs in a uniformly
  shuffled order as the n-token prefix. After CONT, serialize 64 query/answer
  pairs (128 suffix tokens): each query is a key sampled uniformly with
  replacement from the D keys, and its answer is the mapped value. Train
  next-token loss on every suffix position, including query-key positions and
  answer positions. There is no mapping shared across episodes. For task-quality
  evaluation, measure answer top-1 accuracy at answer positions on the
  teacher-forced held-out query/answer suffix.

For the recall task, keys and values are the only ordinary tokens; no separate
query marker is inserted. The position within each query/answer pair identifies
whether a suffix token is a query or answer during the task-quality evaluation.
During the primary sequence-KL evaluation below, suffix tokens are instead
generated autoregressively from the model's full softmax, without forcing the
synthetic query schedule.

### Architecture and training

Train a separate decoder for each task, depth in {2,4}, and seed in
{11,23,37,51,71}, for 20 training runs total. Each model is a causal
GPT-2-style decoder with learned absolute positions, one attention head,
width 64, feed-forward width 256, pre-LayerNorm blocks, GELU, residual
attention/MLP paths, final LayerNorm, dropout 0, and a full softmax over the
task alphabet. The position table has 257 entries. Mixed-length batches are
right-padded with ignored loss targets; causal attention prevents trailing
padding from affecting any real target position. Train with full dense attention and
no cache compression or compression-aware objective.

Use batch size 32, uniformly mix the three prefix lengths, and teacher-force
the full suffix. Use AdamW, peak learning rate 1e-3, weight decay 0.01,
gradient-norm clipping at 1.0, 200 warmup steps, then cosine decay to 0.05 of
peak learning rate. Train exactly 2,000 steps for histogram and 6,000 for
associative recall. Do not select an earlier checkpoint or tune hyperparameters.
Use validation seed 20261011 and 16 validation batches of 32 at the final step.
Fix any implementation-only defaults (including initialization and AdamW
betas/epsilon) in the versioned code/config before dispatch and record them in
the run manifest; they are not tuned after observing evaluation outcomes.

### Frozen quality gates

Apply these gates at the final uncompressed checkpoint:

* Histogram: mean held-out suffix NLL minus the exact Beta(2,2) Bayes suffix
  NLL is at most 0.03 nats/token.
* Associative recall: teacher-forced answer top-1 accuracy is at least 0.90.

Apply these gates separately to each seed. Preserve and report every seed. If a
gate fails, mark that task/depth/seed as gate-failed and exclude it only from
interpretations that presuppose a competent task model; do not retune or replace
the run.

### One-time prefix interventions

For every final checkpoint and each evaluation prefix of length n in
{16,64,128}, prefill the dense model once, then replace only the n prefix KV
records. CONT is outside that compressed prefix and is processed/appended
normally. Thereafter, each model appends its own generated suffix KVs; do not
recompress the cache. Use the same intervention at every layer of the
one-head model.

Compare:

1. **Full cache:** retain all n source records (identity/oracle control).
2. **Uniform weighted subset:** retain m=round(r n) source positions uniformly
   without replacement, for r in {0.25,0.5}; give each retained record weight
   n/m, implemented as attention log-bias log(n/m). The total mass is n.
3. **Positive key/value clusters:** independently at each layer, concatenate
   each record's key and value, choose m centers by deterministic farthest-first
   traversal starting at record zero, and assign records to their nearest
   center in joint key/value Euclidean distance. Assign each chosen center to
   its own cluster to prevent empty clusters under ties. Use each group's key
   mean and value mean; assign its positive cardinality as attention weight. The
   weights sum to n. Keep the original positional contribution already encoded
   in each K/V vector; do not renumber or re-embed a cluster.
4. **Unweighted-subset control:** use exactly the uniform subset indices but
   weight each retained record 1. This intentionally changes total prefix mass
   from n to m and is outside the mass-preserving theorem certificate.

All n are divisible by four, so m is exact for both budgets. Generate paired
subset indices with seed 101 and reuse them across layers and paired method
comparisons for a given held-out prefix and budget. The cluster treatment is
deterministic and does not consume the summary seed. Do not optimize summaries using
suffix tokens, teacher logits, or future-query traces.

### Primary sequence-KL evaluation and sampling

For each task/depth/seed checkpoint, use 16 independently generated held-out
prefixes at each n. Fix the held-out data generator seed to 20261012. For each
prefix and each intervention, sample 8 length-128 continuations from the
intervened model Q using its ordinary full softmax, with no top-k, top-p, or
temperature adjustment. Reuse these same maximum-length paths to form nested
T={8,32,128} estimates; do not draw separate paths for each horizon. This is at
most 16*8=128 paths per n/checkpoint/intervention.

For every sampled Q path and each prefix history, compute the exact conditional
full-softmax KL D_KL(Q_t || P_t), where P is the same checkpoint with its dense
prefix. Feed the same sampled tokens to P and Q at each step, but let P and Q
maintain and append their own suffix KVs at every layer. Sum the conditional
KLs through each nested horizon, then average over the 8 Q paths for a fixed
prefix and over the 16 prefixes. By the KL chain rule, this is an unbiased
Monte Carlo estimator of sequence KL under Q for the fixed held-out prefixes
(the empirical prompt average); it is not a uniform-over-prefix guarantee.
Report the five seed-level means and standard deviations, with per-prefix/path
variation as secondary descriptive information. Do not treat tokens, horizons,
or paths as independent training seeds. Do not estimate sequence TV from these
samples.

The full-cache identity treatment must give zero KL up to numerical tolerance.
Do not teacher-force the synthetic ground-truth suffix for the primary KL
endpoint: the evaluated Q suffix is self-generated, and higher-layer suffix KVs
can differ between P and Q because each model processes the same sampled
history through its own states. Teacher-forced task quality remains a separate
gate/diagnostic.

### Precision and theorem-assumption diagnostics

For both tasks, add a mixture diagnostic only at n=64 with the uniform
weighted-subset budget m/n=0.25. For rho in {0.25,0.5,1}, give each original
prefix record weight 1-rho and each retained summary record weight rho*n/m.
At rho=1, drop the zero-weight original records, leaving m records. At rho<1,
the cache contains n+m records; this is a mixture/precision diagnostic, not a
compression saving. Reuse the same subset indices and held-out prompts as the
corresponding uniform-subset condition.

For each layer and each intervention, compare the dense-prefix and summary
unnormalized moments at queries actually observed along Q's own generated paths:

    E_Z_obs = max_q |Z_hat(q)-Z(q)|/n
    E_N_obs = max_q ||N_hat(q)-N(q)||/n.

Record the query source, layer, horizon, and both errors. These are observed
query maxima only. Do not label them as the uniform all-query epsilon in the
theorem or use them to claim a certificate. In particular, distinguish an
observed empirical curve from a bound holding for every query in the stated
query ball.

Run the deterministic numerical norm audit in
[round6_bounds.py](round6_bounds.py) for each trained checkpoint and retain
per-layer radii/constants, log C_model, and the coarse global epsilon bound.
This conservative audit uses float64 spectral-norm calculations and analytic
LayerNorm bounds; it is not interval arithmetic or a machine-checked proof.
It supplies a global sufficient bound only under its recorded assumptions:
positive mass-n summary weights, K/V within the audited original norm bounds,
unit-mass self-generated suffix records, full softmax, and temperature one.
The unweighted-subset control is outside the certificate. Compare the coarse
KL bound with the separate universal bounded-logit cap T(log(|V|)+2M), where
M bounds every full-model logit's absolute value using the final LayerNorm
radius and output-head norm. If the coarse bound exceeds that cap, label it
looser than the universal cap and retain it in log space. T log(|V|) alone is
not a KL cap for an arbitrary nonuniform P. Do not fit C_model or any gain
factor to measured KL. Full-cache identity has exact zero error.
If no useful global certificate results, report the observed moment diagnostics
and the coarse uniform-norm bound as such; this is not an experiment failure.

Record actual cache tensor storage by method: live K/V and explicit-weight
tensor bytes, per-layer record counts, and estimated attention scratch
separately. Retained indices and method metadata are recorded, but Python host
object overhead and allocator residency are not measured. Original-prefix and
dense-oracle shadow storage are reported separately from Q's tensor storage.
Mixture rho<1 stores n+m records; rho=1 stores m. These are tensor accounting
measurements, not throughput or end-to-end memory savings claims.

## Reporting, failure, and claim boundary

Produce three figures: (1) exact analytic KL versus horizon with the
short-horizon/plateau regimes; (2) exact analytic precision scaling against
epsilon squared; (3) trained multilayer
sequence-KL trajectories with the corresponding observed conditional/moment
diagnostics, stratified by task, depth, seed, and compression method. Include
per-seed points and do not smooth away unsaturated, non-monotone, or failed
cases. Retain the n=64 rho-mixture precision diagnostics for both tasks in
the trained-model aggregates; any supplemental plot must label rho as the
intervention amplitude, rather than a measured uniform-query epsilon.

Stop an affected run for identity, mass, indexing, or software-invariant
failures; record the exact failure and preserve outputs. A failed task-quality
gate is preserved and labelled, with no retuning. Lack of a nonvacuous
all-query constant, failure of a trained model to show the analytic
epsilon-squared curve, or absence of visible long-horizon saturation is a
permitted empirical outcome, not a theorem violation gate: the trained task,
compressor, and finite sampled paths do not instantiate the analytic
worst-case construction or certify the theorem assumptions.

The experiment can illustrate exact worst-case scaling and measure finite
trained-model behavior under named synthetic tasks and one-time prefix
compression. It cannot establish a general trained-transformer theorem,
uniform generation guarantee without a valid all-query certificate, a
minimax space lower bound, repeated-compression stability, or practical
throughput/memory savings. Do not claim NAACL readiness from this protocol or
its eventual results. The supervisor runs Experiment 1 on CPU, then admits
Experiment 2 only when a GPU has at least 4 GiB free and utilization at most
20 percent. Admission is capped at 24 hours and Experiment 2 execution at 12
hours; a timeout preserves partial evidence as incomplete. Keep dispatch receipts private. Preserve the prior
Experiment 0 conclusion NO_HELDOUT_SUPPORT; this protocol does not revise it.

## Primary sources

* Gelberg et al., *Training Transformers for KV Cache Compressibility*,
  [arXiv HTML](https://arxiv.org/html/2605.05971v1), §2, §3 histogram example,
  Appendix C.1 and C.3.
* Li et al., *Transformers as Algorithms: Generalization and Stability in
  In-context Learning*, [PMLR proceedings PDF](https://proceedings.mlr.press/v202/li23l/li23l.pdf),
  §7 and Appendix A.
* Zoology, *Measuring and Improving Recall in Efficient Language Models*,
  [arXiv HTML](https://arxiv.org/html/2312.04927), Appendix E.1–E.2.
