# Incremental review: completed 7B temporal-order pilot

Review base: `01b04cd53590217a8977924b99886445053ec06a` (closed Round 14B delivery).
Evidence head: `344c1d67883cb08d8e28afd87077b4053ac1bada`.
This later metadata-only commit updates this handoff; the evidence head stays fixed.

## Minimal reading order

1. [RESULTS.md](RESULTS.md), new 7B section only: primary table, comparability and limits.
2. [Interpretation](runs/real_llm_temporal_v1_20261012T123114Z/INTERPRETATION.md), then the [small summary](runs/real_llm_temporal_v1_20261012T123114Z/evaluation/summary.json) and [record checks](runs/real_llm_temporal_v1_20261012T123114Z/analysis/record_check.json).
3. [Frozen protocol](experiments/PROTOCOL_REAL_LLM_TEMPORAL_V1.md), [configuration](experiments/real_llm_temporal_v1/config.json), [source identities](runs/real_llm_temporal_v1_20261012T123114Z/FROZEN.json), and the new section of [GPT_CONTEXT.md](GPT_CONTEXT.md).
4. For implementation review: [runner](experiments/real_llm_temporal_v1/run_qwen_probe.py), [CPU checks](experiments/real_llm_temporal_v1/test_cpu.py), and [release verifier](scripts/verify_real_llm_temporal_v1_release.py).

The 8 MB trajectory JSONL is secondary evidence; do not open it first. Earlier
Round 6--14B records, theory and novelty audit do not need rereading for this
update. Frozen source package descriptions retain historical pending-result
language; the completed report supersedes that status. The input ZIP remains
local, identified by SHA-256 in the source manifest.

## New decision-relevant evidence

One fixed Qwen2.5-7B-Instruct checkpoint, twelve original synthetic prompts,
T=130, three policies, both antipodal phases and two Q paths per phase produce
144/144 completed trajectories. Runtime was 516.50 seconds. No training,
model download, prompt removal or rescue sweep occurred.

Equal-prompt mean sequence KL(Q||P) is Block 181.3447, Alternating 185.2281,
Random 189.4147 nats. Direct local attention-error squared is
31.9788/34.8241/35.5113. Block/Alternating aggregate KL ratio is 0.9790, with
Block larger in 6/12 prompts. Paired mean difference is -3.8834 nats;
fixed-template bootstrap95 [-27.9022, 17.1016] spans zero.

Eleven prompts satisfy the prespecified Block/Alternating local max/min <=1.25
tag. Their mean KL is 182.4765/178.3918 (ratio 1.0229), with per-prompt ratios
0.7300--1.2889. None has a twofold gap in either direction. The one
non-comparable prompt remains in the primary aggregate. Thus the earlier
histogram contrast of 11--12x is not reproduced on this pilot.

Plus/minus chooses earlier/later original generated KV at weight two, with
original RoPE positions. It does not orient opposite actual error vectors.
Prompt KV stays exact; each model computes its own future KV. Local error
undoes only the latest edit, holds that layer's query/accumulated Q cache fixed,
sums heads/dimensions and averages layers/events. Matching these pooled
magnitudes does not match vectors, layer sensitivities or history laws.

## Validation and unchanged boundaries

Four focused CPU tests pass. The read-only release verifier checks ten frozen
source hashes, complete grid, per-layer/event errors, exact KL chain sums, and
reproduces the complete summary and frozen bootstrap using temporary copies:

```text
python -B scripts/verify_real_llm_temporal_v1_release.py --input-dir runs/real_llm_temporal_v1_20261012T123114Z
```

This loads no weights and does not regenerate tokens or prove a theorem.
Before dispatch, tiny native/masked and physical/logical parity on CPU/CUDA
BF16 and a six-row 7B smoke passed. The new [7B dependency file](experiments/requirements_real_llm_temporal_v1.txt)
uses Transformers 4.57.1, separate from root dependencies for older experiments.
Staged public files, links, source bytes and privacy exclusions were checked.
Weights, private receipts, machine paths and transient logs are excluded.

The primary metric sums exact full categorical conditional KL on Q-generated
histories and estimates mean schedule-conditional sequence KL; it is not KL
of a hidden-schedule mixture. Two trajectories emit EOS and continue to the
fixed horizon. This measures raw model law, not stop-on-EOS completion quality.
Only two paths per phase and one checkpoint were evaluated. The interval does
not describe a model/task population, and the null contrast is not equivalence
or proof that all temporal effects are absent. All policy KL values are large;
similar risks do not establish fidelity. Logical masks retain allocated KVs;
no task accuracy, physical memory saving or inference speed was measured.

Round 13's finite histogram result and Round 14--14B's negative transfer remain
unchanged. No new Fisher result, compression certificate, space lower bound,
global novelty or publication readiness is claimed. No follow-up was launched.

## Reviewer questions

1. Do the summary and records support the 11/12 comparability count and zero twofold contrasts while retaining all prompts?
2. Are earlier/later retention labels clearly separated from signed error cancellation and the toy intervention?
3. Is the fixed-horizon Q-law KL estimand, including EOS continuation, distinguished from completion quality and mixture KL?
4. Does the bounded negative conclusion avoid claiming equivalence, universal failure or systems benefits?
