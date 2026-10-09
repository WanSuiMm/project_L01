# Review context

## Formal status

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

## Exact question

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

## Source map and invariants

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

## Minimal evidence route

The canonical run is `runs/observable_gain_v1_20261009T190129Z`. Its
`observable_gain_summary.json` is the scientific aggregate;
`execution_summary.json` gives sanitized execution and content hashes;
`summary.json` gives runtime aggregates. Only then inspect `cases.jsonl` for
trajectories, fixed mask indices, seed and features.

Frozen inputs are under `inputs/wikitext_v1_20261009T190129Z`. The input manifest
pins the dataset revision and article list. Weights are not distributed.
Reproduction and offline validation commands are in `README.md`.

## Concrete reviewer questions

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
