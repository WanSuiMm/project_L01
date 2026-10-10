# L01: causal output-relevant KV error

Round 6 studies how one-time prefix-KV replacement affects output distributions, using an exact analytic construction and trained synthetic decoders. The completed formal trained-decoder run is part of the current published snapshot; [RESULTS.md](RESULTS.md) is authoritative for the available evidence, metrics, counts, quality gates, and scientific interpretation.

A completed finite-task result does not establish a general trained-Transformer theorem, a uniform generation guarantee, minimax cache-space lower bound, sequence-TV certificate, or practical inference or memory savings. Treat task-quality failures as part of the reported seed results; do not silently remove them.

## Current reading route

1. [RESULTS.md](RESULTS.md): current published snapshot, canonical aggregates, gates, and claim boundaries.
2. [GPT_CONTEXT.md](GPT_CONTEXT.md): Round 6 scope, source map, evidence route, and reviewer questions.
3. [Frozen Round 6 protocol](experiments/PROTOCOL_ROUND6_TOY_V1.md), then the [analytic config](experiments/configs/round6_scaling_v1.json) and [trained-decoder config](experiments/configs/round6_transformer_v1.json).
4. [THEOREM_AND_PROOF.md](THEOREM_AND_PROOF.md) and [THEORY_STATUS.md](THEORY_STATUS.md) for the user-provided proof draft and its qualifications.

Current endpoint status and numbers belong in RESULTS.md; this reading guide does not duplicate them.

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
