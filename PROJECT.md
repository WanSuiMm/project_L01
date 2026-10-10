# L01: output-relevant causal KV error

Status: **ROUND6_EXECUTION_COMPLETE / ANALYTIC_CONSISTENT /
HISTOGRAM_QUALIFIED / RECALL_TRAINING_UNRELIABLE / GLOBAL_BOUND_UNINFORMATIVE**,
2026-10-10.

The active question is whether one-time, positive mass-preserving prefix-KV
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
