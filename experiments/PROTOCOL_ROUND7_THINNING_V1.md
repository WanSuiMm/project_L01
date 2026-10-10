# L01 Round 7: Equal-Weight Thinning and Autoregressive Law Transfer

Status: frozen experiment specification for an exact analytic study and a
bounded evaluation of existing qualified histogram checkpoints. The analytic
runner does not train models, load checkpoints, or measure GS-Thin. The trained
stage reuses the ten Round 6 histogram checkpoints and performs no new training.

## Question and claim boundary

The analytic question is whether the exact binary histogram decoder realizes
the restricted equal-weight lower bound for every subset of at most m records,
and how its exact sequence KL compares with the balanced uniform-sampling
control. The trained question is what the two directional sequence KL risks
look like for the existing small histogram decoders after a one-time prefix
intervention and self-generated continuations.

The analytic lower bound concerns an equal-weight subset of original records.
It does not lower-bound arbitrary positive weights, synthetic representatives,
or learned encoders. The trained stage measures finite checkpoint, prompt,
compression-seed, and sampled-path cases. It cannot establish a trained-model
exponent, a uniform guarantee over prompts or queries, practical throughput
improvement, or the imported GS-Thin sub-Gaussian guarantee. No result here
establishes publication-level novelty.

The protocol follows the assumptions and distinction between risk directions
in the self-contained Round 7 proof snapshot at
[`THEOREM_AND_PROOF.md`](../theory/archive/round7/THEOREM_AND_PROOF.md).
In particular, compression is performed once, suffix records have unit mass,
and the full and compressed models append their own suffix KVs. KL is averaged
across compression seeds after computing each seed's divergence; a KL involving
a mixture over compression seeds is not an endpoint.

## Analytic experiment

Use n in {32, 64, 128}, and every dyadic m with 1 <= m <= n/2. These values
make n/m a power of two. Use T in {8, 32, 128, 512, 2048}. Exact count-state
dynamic programming evaluates every feasible hard-prefix summary through
T=128. At each (n,m), one feasible minimum-gap witness with s=m,k=0 is also
evaluated exactly at every configured horizon, including T=512 and 2048. Those
long-horizon witness values concern one fixed selector only; they are not
minimax risks and do not by themselves establish long-horizon saturation. For
T=512 and 2048, aggregate minimax hard-prefix risks and balanced
uniform-sampling risks are reported as deterministic curvature envelopes. No
token-path Monte Carlo estimate is used.

Set every key and query to zero. The original length-n hard prefix contains
n/(2m) values +1 and the rest -1, so its mean is mu=1/m-1. A candidate summary
retains s records for some 1 <= s <= m, with k positive records, and gives
each retained record weight n/s. Its mass is n and its mean is 2k/s-1. The
feasible range of k is determined by the two original histogram counts. For
every feasible s,k,

    delta = (2k/s - 1) - mu = 2k/s - 1/m,
    |delta| >= 1/m.

This integer obstruction applies to every subset choice, including a
randomized data-dependent selector. It is the exact analytic hard case; it
does not rely on a particular thinning implementation.

The decoder is the legal one-layer binary histogram construction: attention
outputs the mass-weighted mean of values in {-1,+1}, the binary next-token
probability is sigmoid(a), and each model appends its own generated value with
unit mass. The shared start token has value +1 and is appended before the
first generated token. At generated step t, if J is the number of +1 outputs
among the preceding t-1 generated tokens, the logit for prefix mean nu is

    a_t(nu, J) = (n*nu + 1 + 2*J - (t-1)) / (n+t).

The full model uses nu=mu. A fixed compressed prefix uses its own nu=muhat.
The count-state dynamic program propagates the exact P and Q probabilities for
each suffix count and sums the conditional KL in both directions:
P-to-Q is weighted under P paths, and Q-to-P is weighted under Q paths. This
is exact sequence KL by the chain rule. No binary path enumeration, sampled
continuation, or compression-seed mixture KL is used.

The runner enumerates every feasible s,k for the hard prefix when T is within
the exact-DP cap. It reports the minimum and maximum exact sequence KL over
those summaries in each direction, plus per-summary records. At T=512 and
2048 it reports the analytic lower and upper envelopes

    KL(P||Q), KL(Q||P) >= sigmoid'(1) * S(n,T) / (2*m^2),
    KL(P||Q), KL(Q||P) >= sigmoid'(1) * h(n,T) / (6*m^2),
    KL(P||Q), KL(Q||P) <= S(n,T) * max_feasible(|delta|)^2 / 8,

where S(n,T)=sum_{t=1}^T (n/(n+t))^2 and h(n,T)=nT/(n+T). The first lower
bound uses S directly; the second is the theorem's h/3 form. Bounds are
pathwise for every feasible subset and both KL directions.

The balanced control has n/2 records of each value and samples m positions
uniformly without replacement, with weight n/m on each retained record. If K
is the number of sampled positive records, its exact second moment is

    E[(2K/m-1)^2] = (n-m)/(m(n-1)).

For T<=128, the runner computes the exact hypergeometric average of the two
directional sequence KLs by applying the same count-state DP to every K. At
longer horizons it reports the exact second-moment curvature bounds
sigmoid'(1) * S * E[delta^2]/2 and S * E[delta^2]/8. A full-cache identity
control uses identical prefix means; the runner also evaluates its zero KL by
the count-state DP at the exact-DP horizons and records the analytically known
zero at longer horizons with `exact_dp=false`.

The frozen analytic config has 18 (n,m) choices, five horizons per choice, 90
hard aggregate rows, 90 balanced aggregate rows, and 90 exact minimum-gap
witness rows. Its exhaustive exact-DP horizons are 8, 32, and 128; the witness
DP also reaches 512 and 2048. Primary metrics are the exact minimum hard risk
over feasible equal-weight summaries at short horizons and the exact expected
balanced-control risk at short horizons. The dimensionless h/m^2 and h/m
normalizations are derived from these values and are descriptive; no fitted
constants or empirical exponent gates are used. There is no sampling unit or
confidence interval for these deterministic calculations.

## Existing trained-checkpoint evaluation

The checkpoint source is the completed Round 6 run named
round6_toy_v1_20261010T061030Z. Its frozen config uses training seeds
11, 23, 37, 51, and 71, depths 2 and 4, width 64, and a 257-position context.
The captured summary reports all ten histogram checkpoints trained for 2,000
steps and passing the frozen Bayes NLL gap gate of at most 0.03 nats per token.
Resolve and verify those ten existing checkpoint files from the registered run
manifest before evaluation. Do not train, fine-tune, or replace a checkpoint.
Keep the checkpoint manifest, gate values, and file hashes with the run's
private provenance.

Use the binary histogram task and its existing Beta(2,2) episode generator.
Generate eight new held-out data prefixes at each n in {16, 64, 128}, using
prompt seed 20261013. Reuse those same prompts across checkpoints and methods
for paired comparisons. Each prefix has n data tokens followed by the existing
uncompressed CONT token. Compress only the n data-token KV records once;
process CONT normally. Generate nested suffix horizons T in {8, 32, 128}.
The largest sequence uses 128 data tokens, one CONT token, and 128 generated
tokens, which fits the checkpoint context of 257 positions.

Compare two mass-preserving interventions at each prefix and budget:

* GS-Thin, using `gs_thin_indices(gram, m, rng)` and
  `softmax_kv_gram(keys, values, value_scale=V_l)`. `V_l` is the fixed
  checkpoint/layer value-radius bound from `round6_bounds.audit_model(model)`;
  it is fixed globally for that layer and is not fitted from the observed
  prefix or from KL results. Keys are scaled by `d_model**(-1/4)` when the
  softmax-KV Gram matrix is constructed.
* Uniform sampling without replacement, retaining the same number m of
  original records and assigning each retained record weight n/m.

Use budget fractions {0.125, 0.25, 0.5}. The resulting m values are 2, 4, 8
for n=16; 8, 16, 32 for n=64; and 16, 32, 64 for n=128. Use eight
compression seeds 101 through 108, independently namespaced from the five
training seeds and from generated-path RNGs. Use path seed 20261014 for
continuation sampling. The stable path RNG key is
`("round7-path", path_seed, model_seed, depth, n, prompt_id,
compression_seed)`; method, budget, and direction are deliberately omitted so
the same underlying uniform streams are reused across methods and budgets.
Each combination uses a T-by-32 uniform array: its first 16 streams generate
Q paths and its next 16 generate P paths. Use 16 maximum-length continuation
paths from P and 16 from Q per prompt and compression seed. Reuse each path for
the nested horizons. For P-to-Q, draw paths from P and accumulate exact
conditional KL(P_t||Q_t). For Q-to-P, draw paths from Q and accumulate exact
conditional KL(Q_t||P_t). At each step, feed the same sampled token history to
both models while advancing each model's own KV state and appending its own
suffix record. Use full softmax, with no top-k, top-p, temperature change, or
recompression.

The compression seed is not a training seed. For each depth, training seed is
the primary independent statistical unit. First average paths within
prompt-compression-seed, then average over the eight prompts and eight
compression seeds for each training seed. Report every one of the five
training-seed estimates, their mean and standard deviation, and a 95%
percentile bootstrap interval across those five training-seed estimates using
10,000 resamples and bootstrap seed 20261015. Prompt clusters, compression
seeds, paths, tokens, and horizons are nested observations, not additional
independent training units. Report their variation as secondary descriptive
information. The interval reflects only the five checkpoint training seeds;
it does not provide a prompt-population or arbitrary-query guarantee.

The GS-Thin uncertainty flag is
`FLOAT64_QUARTIC_GS_THIN;_IMPORTED_SG_THEOREM_NOT_EMPIRICALLY_CERTIFIED`.
Store the source revision, config hash, per-checkpoint `log_C_star`, and the
per-row global-risk diagnostic
`log_V_GS = log_C_star + log(h(n,T)) - 2*log(m)`; mark the small-risk condition
only when `V_GS < 1/4`. `C_star` comes from the fixed model-level norm audit,
including each layer's fixed `V_l`; it is not fit to observed KL. The
`observed_reference_numerator_mse` diagnostic uses the observed prefix value
norm as its normalizer and is not a certificate. The Gram walk is implemented
in float64, symmetrizes the input, and clips negative eigenvalues only when
they are within `128 * eps * n * matrix_scale`; larger negative eigenvalues
raise an error. Its singular paired-Gram minimizer uses minimum-norm
`lstsq(rcond=None)` with no ridge. The first iteration's empty newly-bound set
is a no-op. On singular walks with simultaneous boundary hits, all tied hits
are frozen together as an explicit tie-safe extension of the reference
min-one update; this is recorded as an implementation extension, not as a
claim that the imported theorem has been requalified. Empirical sampler draws
do not verify the imported sub-Gaussian theorem. Stop the affected
configuration and preserve partial evidence if a selected-index list has the
wrong length, contains duplicates or out-of-range indices, fails the mass-n
check, or produces non-finite conditional distributions or KL. The full-cache
identity treatment must have zero KL within numerical tolerance. A failed
pre-existing checkpoint quality gate excludes that checkpoint only from
competent-model interpretation; it is retained in the record and is not
retrained.

No trained KL threshold or fitted scaling exponent is a gate. In particular,
do not declare an m^-2 trained-model law from a finite prompt grid. The trained
endpoint is the per-checkpoint, per-method, per-budget, per-prefix-length, and
per-horizon mean sequence KL in each direction, with the five training-seed
values retained.

## Gates, provenance, and interpretation

The analytic gates check the exact one-over-m integer obstruction, both
directional DP risks against their curvature lower bounds, the uniform
hypergeometric second-moment identity and risk envelope, full-cache identity,
finite logits, finite outputs, and DP normalization. Any failure is an
analytic implementation failure; preserve its outputs and do not describe the
gate as a theorem or novelty validation.

The trained software gates check checkpoint identity and quality metadata,
compression cardinality and index validity, positive weights summing to n,
full-cache identity, finite full-softmax distributions, and correct
same-history conditional-KL accumulation with separate model KV updates.
These checks qualify implementation and bookkeeping only. They do not test
the Round 7 theorem or the GS-Thin guarantee.

The analytic runner writes `summary.json`, `hard_equal_weight.csv`,
`hard_equal_weight_exact_cases.csv`,
`hard_minimum_gap_witness_exact.csv`, `uniform_balanced.csv`,
`uniform_balanced_exact_cases.csv`, `identity_controls.csv`, and
`provenance.json` under its requested run directory. The trained runner should retain per-prompt,
per-compression-seed, and per-path records, seed-level directional aggregates,
checkpoint hashes, effective config, and source hashes. Keep large raw records
secondary to compact aggregates.

Interpret analytic hard-prefix results as evidence about the stated restricted
equal-weight subset class. Interpret balanced uniform results as a finite
sampling control whose expected variance is exactly known. Interpret trained
results only as measurements on the ten existing, task-qualified small
checkpoints and named synthetic prefixes. Do not infer performance, arbitrary
positive-weight lower bounds, repeated-compression behavior, or a
uniform-over-query guarantee.

## Reproduction

From the project root, run the CPU analytic experiment into a fresh directory:

    python -B experiments/round7_analytic.py --config experiments/configs/round7_thinning_v1.json --run-dir runs/NEW_RUN/analytic
    python -u experiments/round7_transformer.py --config experiments/configs/round7_thinning_v1.json --checkpoint-root PATH_TO_ROUND6_MODELS --run-dir runs/NEW_RUN/transformer --device cuda
    python -B experiments/analyze_round7.py --config experiments/configs/round7_thinning_v1.json --run-dir runs/NEW_RUN

Replace `PATH_TO_ROUND6_MODELS` with the existing run's `experiment2/models`
directory. Operational host paths remain in private launch receipts. The
trained stage is executed by the server-side evaluation integration after
the ten checkpoint files and their manifest gates are verified. This protocol
does not authorize modifying the frozen Round 6 outputs or launching new
training.
