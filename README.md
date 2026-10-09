# L01: causal output-relevant KV error

Can information available when a KV cache is compressed predict how a persistent
eviction will change future output distributions? The first frozen-model pilot
completed all **144 cases**, but adding four chat-suffix-position proxy scores did
**not** improve heldout prediction over a current-position proxy:
`NO_HELDOUT_SUPPORT` (prediction-MSE reduction **−0.184%**, confidence interval
crossing zero). This is a negative result for this particular proxy, not an
impossibility result for KV compression.

## Start here

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

## Verify the released evidence

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
