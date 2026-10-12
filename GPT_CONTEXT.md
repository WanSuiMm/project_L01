# Review context

## Latest: directional nonlinear remainder attempt

Formal status: `execution=COMPLETE_40_HISTORIES_SAME_ONE_CASE`,
`remainder_feasibility=FAIL_UPPER_30.905579`,
`first_order_diagnostic=UNCHANGED_PASS`, `half_cache=STILL_UNCERTIFIED`.
Read the new section of [RESULTS.md](RESULTS.md), then
[INTERPRETATION.md](runs/round9_directional_remainder_v1_20261012T024528Z/INTERPRETATION.md).
The signed first derivative is preserved, but scalar second-derivative norm
enclosures grow through the second block and final LayerNorm. Old upper 36.6231
becomes 30.9056 after conditional intersections; raw upper is 3.43908e17.
Empty-history upper is already 3.32726. This is a limitation of this enclosure,
not evidence that the measured JVP is inaccurate or every directional method fails.

Sources: `experiments/directional_norm_jets.py:Jet`, `layernorm`, `gelu`,
`attention`, `rollout_bounds`; `experiments/round9_directional_remainder.py:sequence_bound`
uses the full P tree without Q weighting. Derivation:
`theory/ROUND9_DIRECTIONAL_REMAINDER_V1.md`; protocol/config are under
`experiments/`. `scripts/verify_round9_directional_remainder.py:verify` audits
recorded arithmetic/hash bindings. No outward rounding, novel theorem,
multi-case generalization or systems gain is established.

## Round 9 follow-up diagnostic

Read the new section of [RESULTS.md](RESULTS.md), then
[INTERPRETATION.md](runs/round9_directional_diagnostic_v1_20261011T183000Z/INTERPRETATION.md).
Formal status: `diagnostic=COMPLETE_40_HISTORIES_ONE_MODEL_ONE_PREFIX`,
`first_order_screen=PASS_MAX_SCALED_REMAINDER_0.0636924`,
`certificate=HALF_CACHE_STILL_UNCERTIFIED`, `training=NOT_PERFORMED`.
The frozen choice is the first seed and first n64 prompt, not a new model grid.
The measured hidden-radius slack (median 166.4x) substantially exceeds the
additional centered-head norm slack (median 5.75x); final-LN slack at the
actual norm is only median 1.12x. These ratios are separate diagnostics,
not a telescoping decomposition. Actual Q states/masses enter the diagnostic
and cannot be inputs to a P-only certificate. No nonlinear-remainder
supremum or long-horizon result has been established.

Exact symbols: `experiments/round9_directional_diagnostic.py:path_rollout`
implements the weighted-measure path, absolute positions and own-suffix KV;
`analyze_case` measures JVP, endpoint changes and oracle slack substitutions.
The protocol is `experiments/PROTOCOL_ROUND9_DIRECTIONAL_DIAGNOSTIC_V1.md`.
`scripts/verify_round9_directional_diagnostic.py:verify` checks retained
arithmetic/source-copy hashes without model replay. Raw vectors in
`histories.jsonl` are secondary evidence; do not open them before the aggregate.

## Current Round 9 local-bound endpoint

Start with the Round 9 section in [RESULTS.md](RESULTS.md), then the public
snapshot `runs/round9_local_cert_v1_20261011T045942Z/snapshots/20261011T174341Z/`.
Its `summary.json` and `RESULTS.md` give the aggregate; `verification.json` is
the recorded-endpoint audit, and `condition_metrics.csv` contains the compact
per-condition metrics. `checkpoints.json`, `config.json`, `prefixes.json` and
`provenance.json` bind the evaluated models, inputs and sources. Use the
run-level `RUN_STATUS.md` for dispatch and retrieval status. Per-history files
are omitted from the public snapshot and are not the review entry point.

Formal status: `execution=COMPLETE_330_OF_330`,
`primary_gate=FAIL_0_OF_5_MODELS_FOR_EACH_METHOD`,
`training=NOT_PERFORMED`,
`numerical=FLOAT64_ANALYTIC_BOUND_EVALUATION_NOT_INTERVAL_CERTIFIED`.
The five original qualified width-64, two-layer Round 6 histogram models were
evaluated without retraining. The independent unit is the training seed;
prefixes are secondary variation. The endpoint is a finite-horizon bound
evaluation, not an outward-rounded enclosure, theorem falsification, novelty
result, or memory/throughput measurement.

At the primary n=64, m=32 budget, the threshold counts are across 15
seed-prefix rows per method. The gate requires at least two of three prefixes
with upper KL(Q||P) <= 0.1 for at least four of five seeds.

| Method | Upper mean (min/max) | Mean exact KL(Q||P) | <=0.02 | <=0.1 | <1 | Below reference cap |
|---|---:|---:|---:|---:|---:|---:|
| Agglomerative | 8.4763 (range 0.919 to 36.623) | 5.44e-6 | 0/15 | 0/15 | 1/15 | 15/15 |
| GS-Thin | 37.4676 (range 36.525 to 38.139) | 0.003381 | 0/15 | 0/15 | 0/15 | 9/15 |
| Uniform subset | 37.4676 (range 36.525 to 38.139) | 0.02217 | 0/15 | 0/15 | 0/15 | 9/15 |

Being below the reference cap records an improvement over that cap; it does
not establish a high-fidelity bound. Agglomerative at the near-full m=63
control has mean upper 0.0065908, with 14/15 <=0.02 and 15/15 <=0.1; this
retains 63/64 records and does not qualify substantial compression. At the
25% record-reduction budget m=48, agglomerative has mean upper 4.1374, with
10/15 <1 and 0/15 <=0.1. No memory or speed benefit was measured, and record
reduction is not a systems measurement.

| Concept | Exact source symbol |
|---|---|
| Frozen checkpoint identity and quality checks | `experiments/round9_original_models.py:load_original` |
| Reference P tree on the fixed hypothetical histories | `experiments/round9_original_models.py:reference_tree` |
| Prefix summary construction and mass-preserving weights | `experiments/round9_original_models.py:make_summary` |
| P-derived analytic bound plus separate finite Q validation | `experiments/round9_original_models.py:evaluate_summary` |
| Aggregate thresholds and predeclared seed gate | `experiments/round9_original_models.py:aggregate` |
| Local propagation and conditional KL bound | `experiments/round9_certify.py:bound_all_steps`, `certified_conditional_bound` |
| Recorded endpoint audit and compact export | `scripts/analyze_round9_endpoint.py:validate`, `load_rows` |

The endpoint verifier checks recorded source/config/checkpoint bindings, row
coverage, recorded sequence inequalities, full controls and aggregate/gate
arithmetic. It does not replay weights or omitted per-history inequalities,
produce an interval enclosure, or verify theory or novelty. The frozen Round 9
protocol remains the prospective protocol. Do not edit its status sentence;
use the completed run status and snapshot for the endpoint outcome.

## Current Round 7 review context

Start with [RESULTS.md](RESULTS.md), then [GPT_HANDOFF.md](GPT_HANDOFF.md).
Formal status: `execution=COMPLETE`, `analytic=ANALYTIC_GATES_PASS`,
`trained=HISTOGRAM_QUALIFIED`, `empirical=GS_USUALLY_LOWER_KL`,
`certificate=GLOBAL_SMALL_RISK_UNCERTIFIED`, `novelty=UNRESOLVED`.
Round 6 and Experiment 0 below retain their own estimands and uncertainty methods.

Round 7 freezes one-time prefix compression, positive equal weights n/m,
uncompressed CONT, and unit-weight self-generated suffix records. GS-Thin and
uniform subsets retain original records; the lower-bound class excludes
arbitrary positive weights, synthetic representatives and learned encoders.
The archived proof is a conditional draft using an imported SG interface.
The tie-safe float64 implementation and its software tests do not certify
that imported theorem. Historical novelty and verification files in the archive
are user-supplied context, not independent certification by this experiment.

The trained endpoints are E_R KL(Q_R||P) and E_R KL(P||Q_R). Each direction
uses its own sampled histories; both shadows receive the same tokens and append
their own KV records. Risk is averaged over compression seeds after KL is
computed, rather than taking KL of a mixture. Five training seeds per depth
are the independent units, with nested prompt/draw/path averaging and paired
seed bootstraps. The 108 cell intervals are not multiplicity-adjusted.

| Concept | Exact source symbol |
|---|---|
| Recursive GS halving / fixed-radius softmax-KV kernel | `experiments/gs_thin.py:gs_thin_indices`, `softmax_kv_gram` |
| Equal-mass prefix replacement | `experiments/round7_transformer.py:select_summary` |
| Both directional KL laws, same-history replay, own suffix append | `experiments/round7_transformer.py:evaluate_batch` |
| Existing checkpoint quality / norm audit / evaluation grid | `experiments/round7_transformer.py:run` |
| Exact hard-class DP, uniform hypergeometric control and witness | `experiments/round7_analytic.py:sequence_kl_dp`, `run_analysis` |
| Raw key / direction coverage and full identity checks | `experiments/analyze_round7.py:_validate_rows` |
| Seed means and paired bootstrap | `experiments/analyze_round7.py:_aggregate_primary` |
| Global certificate separate from observed KL | `experiments/analyze_round7.py:_global_certificate_report` |
| Public hash and seed-level replay checks | `scripts/verify_round7_release.py` |

The canonical compact endpoint is
`runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/`.
Its `analysis/RESULTS.md` and figures are the short route; `analysis/aggregates.json`
retains all seed estimates and paired comparisons. `analytic/` contains all
analytic CSVs; `checkpoints.json` contains hashes, quality and global constants.
`publication_manifest.json` binds frozen sources and public payloads with
sanitized execution timing. Raw trained rows and weights remain private.
Reported raw completeness is server validation; public verification only
reconstructs the compact seed-level endpoint and hash bindings.

At short horizons the analytic class minima are exact. Longer hard/uniform
aggregate rows are explicitly labeled curvature envelopes; the all-horizon
exact witness is not a minimax endpoint. Do not turn the finite trained curves
into an m^-2 theorem, a general NLP result, a uniform query guarantee or
practical inference savings. No global forward small-risk condition is met.

Focused review questions:

1. Does the theorem's imported SG interface apply to this float64 recursive
   implementation with its explicitly documented simultaneous-boundary rule?
2. What do the trained horizon trends explain beyond this histogram task,
   given the n=16,m=2 failures and the vacuous global certificate?
3. Does the restricted equal-weight class lower bound provide enough value
   when arbitrary weights/representatives and useful constants remain open?

## Current Round 6 review context

The completed formal trained-decoder run is part of the current published snapshot. Start with [RESULTS.md](RESULTS.md) for its current metrics, counts, task-quality gates, and claim boundary. Treat a running or partial group as incomplete, and do not infer an endpoint from training progress.

Round 6 has two distinct parts: an exact analytic two-atom prefix-replacement construction, and finite trained synthetic-task decoders for histogram prediction and associative recall. The analytic construction is not a learned-model result. The trained study does not establish a general Transformer theorem, uniform-over-prefix guarantee, minimax space lower bound, or sequence-TV endpoint.

For the trained study, the independent unit is the training seed. Report the full seed set, including seeds that fail frozen task-quality gates; do not filter failures out of the primary estimate. Prefixes are secondary clusters and generated continuations are Monte Carlo replicates. The primary KL sums conditional D_KL(Q_t || P_t) along continuations sampled from Q, with P replaying the same tokens through its own dense cache. It estimates sequence KL for the fixed empirical prompt set, not a uniform-prefix guarantee.

Observed-query moment errors and empirical epsilon values are diagnostics on sampled queries, not uniform-query certificates. The rho<1 precision mixtures store n+m prefix records, and the unweighted-subset control changes total mass, so neither should be described as a mass-preserving m-record compressor. Tensor-byte accounting is not throughput or end-to-end memory savings.

## Round 6 source map

| Concept | Exact source symbol |
|---|---|
| Exact analytic construction and scaling rows | `experiments/round6_scaling.py:_case_rows`, `bernoulli_kl_logit_to_half` |
| One-time prefix-cache replacement and controls | `experiments/round6_transformer.py:compress_prefix_cache` |
| Q-sampled continuation and conditional KL | `experiments/round6_transformer.py:evaluate_prompt`, `conditional_kl` |
| Task training and frozen final quality metrics | `experiments/round6_transformer.py:train_one_model`, `run_experiment` |
| Global norm audit and theorem-bound qualification | `experiments/round6_bounds.py:audit_model`, `experiments/analyze_round6.py:_derive_bound_status` |
| Seed-level aggregation and incomplete-run exclusion | `experiments/analyze_round6.py:_aggregate_exp2_groups`, `_load_exp2` |

Source maps above identify implementation, not independent proof verification. The norm audit is a deterministic numerical audit under recorded assumptions, not interval arithmetic or a machine-checked proof.

## Minimal Round 6 evidence route

Use [RESULTS.md](RESULTS.md) first. The canonical run is `runs/round6_toy_v1_20261010T061030Z`; the completed endpoint is its `snapshots/20261010T073818Z/` directory. Within that snapshot, `analysis/RESULTS.md` is the derived report, `analysis/aggregates.json` has curves and bounds, `scientific_summary.json` has quality/coverage records, and `provenance.json` binds frozen sources and public endpoints. The run's top-level `analysis/` is a preserved earlier Experiment-1-only snapshot. The analytic source summary is `experiment1/summary.json`. Large per-prompt JSONL and operational files are secondary and should not be the entry point.

The frozen protocol is [experiments/PROTOCOL_ROUND6_TOY_V1.md](experiments/PROTOCOL_ROUND6_TOY_V1.md). The completed endpoint counts and all current metrics are centralized in RESULTS.md.

## Concrete Round 6 reviewer questions

1. Does the implementation replace only the prefix cache once, append CONT and suffix records correctly, and compute the stated horizon sum of Q-to-P conditional KL on Q-sampled paths?
2. Which findings come from the exact finite analytic curves or sampled trained tasks, and do any reported norm bounds actually satisfy the theorem assumptions with a nonvacuous uniform-query guarantee?
3. Are the frozen task-quality gates and full training-seed coverage reported, including failures, with seeds as the primary uncertainty unit and prefixes/paths kept secondary?

## Historical: Experiment 0 frozen-model pilot

The following review context applies only to the frozen Experiment 0 pilot; its status does not describe Round 6.

### Formal status

- `execution_status`: COMPLETE, 144/144 cases.
- `scientific_verdict`: NO_HELDOUT_SUPPORT.
- `scope`: frozen-model mechanism pilot, not a compression benchmark.
- `theory_status`: conditional identities and sufficient conditions; no
  computable, non-vacuous Transformer certificate established.
- `novelty_status`: unresolved; output-KL scoring and future-query geometry
  have close prior work.

Start with `RESULTS.md`, then the frozen protocol. Do not start with JSONL,
historical theory drafts, operational tooling, or model/data caches. No server
access is needed for offline review.

### Exact question

Does adding the mean and maximum of four already observed query-position proxy
KL values improve heldout prediction of future KL over a current-position proxy?
The answer in this run is no. Calibration uses 8 articles; heldout uses 16
disjoint articles. Six cells per article are dependent observations.

The candidate is a standardized ridge regression with current/mean/max log-KL
features. The baseline uses current log-KL alone. Both predict log10 of future
cumulative conditional KL, clamped at 1e-10. The primary gate and article-level
bootstrap were frozen before execution. No outcome-based retuning is included.
The four positions are the suffix of the chat-templated prefix, covering
boundary/header queries rather than the last four article-content queries.
The bootstrap conditions on the sampled paths and measures article variation;
one rollout per cell does not quantify additional sampling variance.

### Source map and invariants

| Concept | Exact source symbol |
|---|---|
| Article-disjoint public input freezing | `prepare_wikitext_prefixes.py:main` |
| Tokenization preserves chat boundaries | `observable_gain.py:_tokenize_prompt` |
| One-layer persistent deletion | `_PersistentMaskHook` |
| Budget-matched random / attention masks | `_choose_mask_positions` |
| Direct-residual proxy and final RMSNorm derivative | `_proxy_features`, `_rmsnorm_jvp` |
| Same-prefix Q/P replay and future positions | `_rollout_case` |
| Q || P KL computation | `_kl_q_to_p` |
| Calibration-only fit | `analyze_observable_gain.py:_fit_ridge`, `analyze` |
| Paired article uncertainty | `_prompt_mse`, `_bootstrap_interval` |
| Evidence replay without editing frozen data | `scripts/verify_results.py` |

Source modules above are in `experiments/` unless another path is given. The
attention hook is disabled for P forwards. Q and P each mutate separate cache
copies, and P uses the same Q-sampled tokens. Earlier cached states preceding the
four-position intervention are shared; later states evolve under the edit.

The proxy uses pre-eviction full-cache attention and value statistics, projects
the finite deletion-induced attention difference through the output projection,
then through the final RMSNorm derivative and LM head. It keeps the direct
residual path and omits later attention/MLP responses. It is not a full causal
Jacobian, a Fisher certificate, or the published BPKV algorithm.

### Minimal evidence route

The canonical run is `runs/observable_gain_v1_20261009T190129Z`. Its
`observable_gain_summary.json` is the scientific aggregate;
`execution_summary.json` gives sanitized execution and content hashes;
`summary.json` gives runtime aggregates. Only then inspect `cases.jsonl` for
trajectories, fixed mask indices, seed and features.

Frozen inputs are under `inputs/wikitext_v1_20261009T190129Z`. The input manifest
pins the dataset revision and article list. Weights are not distributed.
Reproduction and offline validation commands are in `README.md`.

### Concrete reviewer questions

1. Does the implemented intervention and t=1..32 target match the frozen
   protocol, and is the complete-cache P reference kept unmodified?
2. Are fit isolation and article-level uncertainty sufficient for the stated
   pilot verdict, without treating 96 heldout cells as independent samples?
3. Do template-bound recent positions and the direct-residual approximation
   sufficiently explain the narrow scope of the negative result?
4. What genuinely different, past-only mechanism would justify another run,
   beyond adding more recent KL summaries or a retrospective future Gramian?

Historical drafts are preserved in `theory/archive/` for context. Their stronger
phrasing and bibliographic errors are qualified in `THEORY_STATUS.md`; their
synthetic checks are not this experiment's scientific evidence. This repository
is self-contained and requires no sibling research project.
