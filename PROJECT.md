# L01: output-relevant causal KV error

Status: **ROUND9_COMPLETE / HALF_CACHE_FEASIBILITY_FAIL / NEAR_FULL_CLUSTER_ONLY**;
previous **ROUND7_EXECUTION_COMPLETE / GS_EMPIRICAL_ADVANTAGE / GLOBAL_CERTIFICATE_VACUOUS**;
previous **ROUND6_EXECUTION_COMPLETE / ANALYTIC_CONSISTENT /
HISTOGRAM_QUALIFIED / RECALL_TRAINING_UNRELIABLE / GLOBAL_BOUND_UNINFORMATIVE**,
2026-10-12.

Round 9 launched reference-only finite-tree local-bound evaluation on the five
original qualified two-layer histogram checkpoints. The frozen
[protocol](experiments/PROTOCOL_ROUND9_LOCAL_CERT_V1.md) specifies 330 conditions:
three fixed prefixes at n=16 and 64, T=4 over the full three-token tree,
near-full and 25%/50%/75% old-prefix record reductions, with positive cluster,
uniform subset and eligible GS-Thin summaries. Eight local software tests pass.
The formal CPU run completed all 330 conditions on 2026-10-12 at 01:43:41
Asia/Shanghai in 67.78 seconds. All five checkpoint hashes and recorded quality
results matched; no training occurred. At n=64,m=32 all methods have 0/5 passing
models and 0/15 prefixes with upper KL<=0.1. Agglomerative mean upper KL is
8.4763 versus actual KL 5.4366e-6; its bound improves on the trivial cap but
does not meet the frozen high-fidelity gate. Only near-full agglomerative
n=64,m=63 has 15/15 upper KL<=0.1. Read [RESULTS.md](RESULTS.md), then the
[completed run status](runs/round9_local_cert_v1_20261011T045942Z/RUN_STATUS.md).
The compact snapshot retains all 330 condition metrics, with independently
checked coverage, seed/group arithmetic and gates. Earlier failed connections
and dispatch records remain private. Round 6/7 results are unchanged. These
are float64 analytic-bound evaluations, not strict interval certificates.

Canonical entry commands (repository root):

```text
python -B -m unittest discover -s experiments -p 'test_round9*.py' -v
python -B experiments/round9_original_models.py --config experiments/configs/round9_local_cert_v1.json --checkpoint-root PATH_TO_ROUND6_MODELS --preflight
python -B experiments/round9_dispatch.py smoke --source-dir experiments --checkpoint-root PATH_TO_ROUND6_MODELS --run-dir PRIVATE_NEW_SMOKE
python -B experiments/round9_dispatch.py launch --source-dir experiments --checkpoint-root PATH_TO_ROUND6_MODELS --run-dir PRIVATE_NEW_RUN
python -B scripts/analyze_round9_endpoint.py --input-dir runs/round9_local_cert_v1_20261011T045942Z/snapshots/20261011T174341Z
```

The dispatcher records a durable CPU process receipt and enforces a one-hour
limit. It performs no training and never substitutes the separate width-16
Round 9 attachment fixture for an original checkpoint.

Round 7 completed on 2026-10-10 at 11:21:13 UTC in 2,776.51 seconds. It evaluates
float64 GS-Thin against mass-preserving uniform subsets, reusing all ten
qualified Round 6 histogram checkpoints without training. The question is how
prefix record budget relates to both directional sequence KL risks after
one-time compression and each decoder's own suffix-KV updates. The candidate
worst-case rate is `n*T/(m^2*(n+T))` in the restricted equal-weight subset class;
it is not a required exponent on trained models. Read the frozen
[Round 7 protocol](experiments/PROTOCOL_ROUND7_THINNING_V1.md),
[proof snapshot](theory/archive/round7/THEOREM_AND_PROOF.md), and
[run status](runs/round7_thinning_v1_20261010T102523Z/RUN_STATUS.md).
All 70,560 records passed server-side validation; all ten checkpoints passed
their frozen quality gate. GS-Thin mean KL is lower in 96/108 paired cells;
all 12 exceptions use n=16,m=2. No GS forward row meets the audited small-risk
condition (0/17,280). These are finite histogram results, not a general theorem
or useful uniform certificate. Round 6 and Experiment 0 evidence remain unchanged.

The completed compact endpoint is
`runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/`.
Start with [RESULTS.md](RESULTS.md). Reproduction commands are in
[README.md](README.md); offline publication verification is
`python -B scripts/verify_round7_release.py`. The public snapshot retains all
training-seed aggregate values and analytic CSVs, but excludes raw trained
rows, checkpoints and operational records. Its verifier does not reconstruct
row-level coverage or recalculate checkpoint norms.

The completed Round 6 question was whether one-time, positive mass-preserving prefix-KV
compression explains sequence KL through the candidate scale
`epsilon^2 * n*T/(n+T)` in trained dense causal decoders. Read
[THEOREM_AND_PROOF.md](THEOREM_AND_PROOF.md), the frozen
[Round 6 protocol](experiments/PROTOCOL_ROUND6_TOY_V1.md), then
[the run status](runs/round6_toy_v1_20261010T061030Z/RUN_STATUS.md).

The run is `runs/round6_toy_v1_20261010T061030Z`. Experiment 1 completed all
160 analytic configurations and all recorded gates passed. Experiment 2
completed 20 models, 22,080 prompt rows and 276 groups in about 63 minutes.
All 10 histogram models passed the frozen training gate; only 3/10 recall
models passed. All seeds remain in the primary aggregates. The global norm
bound was looser than the universal bounded-logit cap in every applicable
group-seed case. Read [RESULTS.md](RESULTS.md) for interpretation.

The completed evidence is in `snapshots/20261010T073818Z/` under this run.
Its `analysis/` contains all three figure pairs. The earlier top-level
`analysis/` is preserved as an Experiment-1-only launch-time snapshot.

The proof draft references historical verification and novelty files absent
from this checkout. The new experiment does not claim to have reproduced those
historical checks or established publication-level novelty.

Reproduce from this repository root, using a fresh output directory:

```text
python -B experiments/round6_scaling.py --config experiments/configs/round6_scaling_v1.json --run-dir runs/NEW_RUN/experiment1
python -u experiments/round6_transformer.py --config experiments/configs/round6_transformer_v1.json --run-dir runs/NEW_RUN/experiment2 --device cuda
python -B experiments/analyze_round6.py --run-dir runs/NEW_RUN --transformer-config experiments/configs/round6_transformer_v1.json
```

The registered server runtime used Python 3.12.13, PyTorch 2.11.0+cu128,
NumPy 2.3.5 and Matplotlib 3.10.9. Model and task definitions are in
`experiments/round6_transformer.py`; `experiments/round6_bounds.py`
audits conservative global constants. Source/configuration hashes are in the
run's sanitized provenance; operational receipts and checkpoints remain private.

## Previous Experiment 0: frozen negative result

Status: **EXPERIMENT_0_COMPLETE / NO_HELDOUT_SUPPORT**.

Formal run: `runs/observable_gain_v1_20261009T190129Z`. All 144 cases completed in
157.13 seconds including startup and analysis. Adding recent-position scores
changed heldout prediction MSE from 0.1040362242 to 0.1042276286 (−0.184%
improvement); the paired article-bootstrap interval crosses zero. Read
[RESULTS.md](RESULTS.md) for the frozen gate and interpretation. Source,
configuration and input hashes are checked by `scripts/verify_results.py`.

The GPU smoke completed one middle-layer intervention and two future positions.
No-mask replay probability difference was zero; hard-mask attention-output
reconstruction had relative error 0.0018914. These are software checks, not a
scientific endpoint. The first smoke attempt failed during CUDA allocator
initialization; its source and evidence remain separate from the successful run.

Question: can statistics available at compression time predict the future
conditional-distribution cost of a persistent, structured KV eviction better
than a current-position output-KL proxy?

This is a bounded mechanism pilot on one frozen model, not a compression system
benchmark or a uniform stability certificate. Historical theory snapshots are
included under `theory/archive`; the current qualifications are in
[THEORY_STATUS.md](THEORY_STATUS.md). This repository is self-contained.

Read `experiments/PROTOCOL_OBSERVABLE_GAIN_V1.md`, then the frozen configuration
`experiments/configs/observable_gain_v1.json`. The runner is
`experiments/observable_gain.py`; the analysis entry is
`experiments/analyze_observable_gain.py`.

Run offline verification from this project root:

```text
python -B -m unittest discover -s experiments -p test_observable_analysis.py -v
python -B scripts/verify_results.py
```

GPU reproduction commands and dependencies are in [README.md](README.md).
Model weights are not downloaded by the runner. Runtime paths and dispatch
receipts are private operational records, separate from scientific summaries.
