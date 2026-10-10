# Review context

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
