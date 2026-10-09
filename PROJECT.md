# L01: output-relevant causal KV error

Status: **EXPERIMENT_0_COMPLETE / NO_HELDOUT_SUPPORT**, 2026-10-10.

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
