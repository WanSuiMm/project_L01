# L01 — real-LLM temporal KV risk pilot v1

Use the [canonical v1 protocol](../PROTOCOL_REAL_LLM_TEMPORAL_V1.md) and
[configuration](config.json). This revision fixes local-error normalization,
separates schedule/sample RNGs, validates complete output grids, preserves FP32
score biases, and strengthens tiny-model parity checks. The original supplied
ZIP is preserved. Earlier Round 14B negative conclusions remain unchanged.

From the project root, use the isolated pinned environment:

```text
python -B -m unittest discover -s experiments/real_llm_temporal_v1 -p test_cpu.py -v
python experiments/real_llm_temporal_v1/run_qwen_probe.py --hf-smoke-tiny
python experiments/real_llm_temporal_v1/run_qwen_probe.py --model-path MODEL_SNAPSHOT --out-dir runs/NEW_REAL_LLM_RUN
```

Defaults implement the frozen 12-prompt, 144-trajectory pilot. Do not reuse an
existing output directory. Analysis requires a completed manifest and full grid.
Primary evidence concerns schedule-dependent model-law KL; earlier/later signs
do not imply opposite error vectors. No task-accuracy or systems-saving claim.

## Original package description

This is a **ready-to-run experimental harness and frozen-style protocol**, not a successful large-LLM experiment. This package was prepared in a CPU-only environment without Transformers/model checkpoints. Read `PROTOCOL.md` before interpreting any output.

Main entry: `run_qwen_probe.py` is pinned to `transformers==4.57.1` and standard Qwen2 models using eager attention, including Qwen2.5-7B-Instruct (28 layers, 28 query heads / 4 KV heads). Both arms keep an uncompressed source prompt. Only pairs of **generated tokens** are logically compressed by additive attention masks equivalent to weight-2 original-record retention.

The core protocol checks the important scientific distinction: per-event direct local attention displacement vs cumulative full-softmax `KL(Q||P)` on Q-generated histories. All prompts and schedules must be evaluated, including negative results and failures of the local-matching screen. No physical cache saving or throughput claim.

Prerequisites: `torch` installed with working GPU CUDA, Qwen2.5 model weights downloaded locally (or remote access available to HF), `transformers==4.57.1`, `safetensors`, `accelerate`. Test commands are in `PROTOCOL.md`.

To analyze a completed run without a GPU: `python run_qwen_probe.py --analyze-only --out-dir runs/frozen_pilot_new` (only Python standard library needed for analysis).

**Status:** only CPU algebraic self-test run. Real 7B result is *not yet observed*.
