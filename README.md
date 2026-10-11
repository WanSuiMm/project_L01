# L01: causal output-relevant KV error

Round 9 asks whether local analytic bounds certify substantial prefix-KV compression on the original five two-layer histogram models. All 330 conditions completed: at half cache (n=64,m=32), no method passes the frozen KL<=0.1 feasibility gate. Agglomerative actual KL is tiny, but its mean upper bound is 8.4763; near-full n=64,m=63 succeeds. [RESULTS.md](RESULTS.md) distinguishes these negative and positive findings. Earlier Round 7, Round 6 and Experiment 0 results are preserved.

A completed finite-task result does not establish a general trained-Transformer theorem, a uniform generation guarantee, minimax cache-space lower bound, sequence-TV certificate, or practical inference or memory savings. Treat task-quality failures as part of the reported seed results; do not silently remove them.

## Current reading route

1. [RESULTS.md](RESULTS.md): current Round 9 verdict, half-cache table and near-full control.
2. [GPT_HANDOFF.md](GPT_HANDOFF.md), then [GPT_CONTEXT.md](GPT_CONTEXT.md): incremental review and exact source symbols.
3. [Completed run status](runs/round9_local_cert_v1_20261011T045942Z/RUN_STATUS.md), [verification](runs/round9_local_cert_v1_20261011T045942Z/snapshots/20261011T174341Z/verification.json) and [condition metrics](runs/round9_local_cert_v1_20261011T045942Z/snapshots/20261011T174341Z/condition_metrics.csv): coverage, all seed/prefix cases and audit limits.
4. [Frozen protocol](experiments/PROTOCOL_ROUND9_LOCAL_CERT_V1.md) and [runner](experiments/round9_original_models.py): reference-only bound construction and independent Q-tree measurement.

Verify the compact Round 9 endpoint without model weights:

```text
python -B scripts/analyze_round9_endpoint.py --input-dir runs/round9_local_cert_v1_20261011T045942Z/snapshots/20261011T174341Z
```

The snapshot contains float64 analytic-bound evaluations, not strict interval
certificates. It excludes checkpoints, per-history trees and operational records.
Current reproduction commands are in [PROJECT.md](PROJECT.md).

The completed Round 7 reading route remains below.

1. [RESULTS.md](RESULTS.md): current published snapshot, canonical aggregates, gates, and claim boundaries.
2. [GPT_HANDOFF.md](GPT_HANDOFF.md): changes since the Round 6 review, evidence commit and focused reviewer questions.
3. [GPT_CONTEXT.md](GPT_CONTEXT.md), [frozen Round 7 protocol](experiments/PROTOCOL_ROUND7_THINNING_V1.md), and [configuration](experiments/configs/round7_thinning_v1.json): estimands, source map and controls.
4. [THEORY_STATUS.md](THEORY_STATUS.md), then the [archived Round 7 proof draft](theory/archive/round7/THEOREM_AND_PROOF.md): conditional mathematics and restricted lower-bound class.

Current endpoint status and numbers belong in RESULTS.md; this reading guide does not duplicate them.

## Round 7 verification and reproduction

From this repository root, with [requirements.txt](requirements.txt) installed:

```text
python -B -m unittest discover -s experiments -p test_gs_thin.py -v
python -B -m unittest discover -s experiments -p test_round7_transformer.py -v
python -B scripts/verify_round7_release.py
```

The release verifier checks frozen source and payload hashes, analytic output
hashes, checkpoint metadata, training-seed means and paired bootstrap intervals.
Raw trained rows and model weights are excluded; row-level completeness is a
recorded server validation, not independently reconstructed by this verifier.
It does not certify GS-Thin's imported theorem or recompute checkpoint norms.

Reproduce into a fresh output directory:

```text
python -B experiments/round7_analytic.py --config experiments/configs/round7_thinning_v1.json --run-dir runs/NEW_RUN/analytic
python -u experiments/round7_transformer.py --config experiments/configs/round7_thinning_v1.json --checkpoint-root PATH_TO_ROUND6_MODELS --run-dir runs/NEW_RUN/transformer --device cuda
python -B experiments/analyze_round7.py --config experiments/configs/round7_thinning_v1.json --run-dir runs/NEW_RUN
```

`PATH_TO_ROUND6_MODELS` is the original Round 6 `experiment2/models` directory;
the expected ten checkpoint hashes and quality records are in the
[completed snapshot](runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/checkpoints.json).
Weights are not distributed. Round 6 reproduction below can train replacement
models under the original protocol, but differing checkpoint bytes constitute
a new realization, not exact replay of this endpoint. The Round 7 evaluation
performs no training. GPU execution is needed only for new trained-decoder
measurements, not for reading or verifying the public compact results.

## Round 6 reproduction

Run commands from this standalone repository root. Install the dependencies in [requirements.txt](requirements.txt). The test command checks the frozen analytic, trained-decoder, and analysis code:

```text
python -B -m unittest discover -s experiments -p 'test_round6*.py' -v
python -B -m unittest discover -s scripts -p test_round6_snapshot.py -v
python -B scripts/verify_round6_release.py
```

The release verifier checks source/configuration and result hashes, complete
coverage, per-prompt-to-seed aggregates and full-cache identity. It also
recomputes the exact analytic grid. It does not retrain models or recalculate
checkpoint norms; the published norm audits are recorded numerical results.

To reproduce into a fresh run directory:

```text
python -B experiments/round6_scaling.py --config experiments/configs/round6_scaling_v1.json --run-dir runs/round6_reproduction_v1/experiment1
python -u experiments/round6_transformer.py --config experiments/configs/round6_transformer_v1.json --run-dir runs/round6_reproduction_v1/experiment2 --device cuda
python -B experiments/analyze_round6.py --run-dir runs/round6_reproduction_v1 --transformer-config experiments/configs/round6_transformer_v1.json
```

The analysis includes the trained-decoder endpoint only for a completed formal non-smoke run. A running or partial group is not a completed endpoint. The primary uncertainty unit is training seed; prefixes and generated paths are secondary variation. The unweighted-subset control does not preserve mass, and rho-mixture diagnostics with rho below one store n+m prefix records. Neither is evidence of a mass-preserving m-record compressor.

## Historical: Experiment 0 frozen-model negative result

Can information available when a KV cache is compressed predict how a persistent
eviction will change future output distributions? The first frozen-model pilot
completed all **144 cases**, but adding four chat-suffix-position proxy scores did
**not** improve heldout prediction over a current-position proxy:
`NO_HELDOUT_SUPPORT` (prediction-MSE reduction **−0.184%**, confidence interval
crossing zero). This is a negative result for this particular proxy, not an
impossibility result for KV compression.

### E0 reading route

1. [RESULTS.md](RESULTS.md): canonical results and scientific interpretation.
2. [GPT_CONTEXT.md](GPT_CONTEXT.md): exact scope, source map and reviewer questions.
3. [Frozen protocol](experiments/PROTOCOL_OBSERVABLE_GAIN_V1.md) and
   [configuration](experiments/configs/observable_gain_v1.json).
4. [THEORY_STATUS.md](THEORY_STATUS.md): conditional mathematics, limitations and
   historical drafts. Read this before the theory archive.

The model is Qwen2.5-7B-Instruct; inputs are 24 article-disjoint WikiText prefixes
(8 calibration, 16 heldout). Each prefix has three layer positions and two
budget-matched eviction masks, followed by 32 future prediction positions.
Hard attention masking preserves the behavior of deletion but retains cache
storage. This repository contains **no measured memory saving, inference
speedup, full-sequence TV certificate, or established new compression algorithm**.

### E0 verification

Python 3.12 with NumPy and SciPy is sufficient for offline verification:

```text
python -B -m unittest discover -s experiments -p test_observable_analysis.py -v
python -B scripts/verify_results.py
```

The verifier reconstructs the analysis in a temporary directory and preserves
the frozen results. Do not open the 144-row JSONL first; the concise aggregate
and reading route above provide the necessary context.

For GPU reproduction, install [requirements.txt](requirements.txt), acquire
`Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28` separately, and run:

```text
python experiments/observable_gain.py --config experiments/configs/observable_gain_v1.json --prompts inputs/wikitext_v1_20261009T190129Z/prompts.jsonl --model-path MODEL_SNAPSHOT --run-dir runs/reproduction_v1
python experiments/analyze_observable_gain.py --run-dir runs/reproduction_v1
```

The runner uses existing local weights only. Floating-point results can vary
across hardware and software; published hashes identify the source, inputs and
result files. The revision is recorded in the launch configuration; model and
tokenizer bytes were not independently hashed. Frozen input texts and their
dataset provenance are included.

The corpus is derived from
[Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext);
see [input attribution](inputs/README.md) for its separate licensing and article
titles. Runtime launch records, host paths, model weights and caches are excluded
from public delivery.
