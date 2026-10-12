# L01 real-LLM temporal-order pilot v1

Authorized 2026-10-12 after the user-provided pilot package review. This is a new
bounded external-validity pilot. Round 14B remains closed with its negative
conclusion; this protocol does not reopen or replace its gates.

## Question and claim

On a fixed pretrained Qwen checkpoint, can different orders of earlier/later
KV retention produce different autoregressive generation risk while the logical
cache budgets and pooled direct local attention-error magnitudes are comparable?

The signs mean **retain earlier/later records**, not opposite directions of real
attention errors. A positive contrast supports schedule-dependent risk under
this selection rule. It does not isolate temporal error cancellation, establish
a universal covariance law, or imply task-accuracy or systems benefits.

## Frozen inputs and intervention

- Model: `Qwen/Qwen2.5-7B-Instruct`, cached snapshot
  `a09a35458c702b33eeacc393d103063234e8bc28`; no training or model download.
- Dependencies: Transformers 4.57.1, huggingface-hub 0.36.0 in an isolated
  overlay of the registered CUDA environment. BF16 weights/KV, eager attention,
  FP32 additive score masks, full-vocabulary sampling softmax at temperature 1.
- All twelve prompts in `real_llm_temporal_v1/prompts.json`, unchanged from the
  supplied package. Six Chinese and six English prompts cover three categories.
- T=130 output tokens, 64 post-pair edits. The first edit of y1/y2 affects y4;
  the last edit of y127/y128 affects y130. Original prompt records stay exact.
- Each edit keeps one original atom at weight 2 and masks the other. Positions
  retain their original RoPE indices. Each stream appends its own future KV.
- Block: 32 earlier then 32 later; Alternating: earlier/later alternating;
  Balanced Random: 32 of each in a seeded permutation. Both antipodal phases
  and two sampling replicates per prompt; 144 total trajectory records.
- Base seed 20261012. Separate domain seeds for random schedules and token
  uniforms. Common token uniforms couple policies/phases within prompt/replicate.
  This couples paths without making their histories identical.
- Model, prompt/config and source identities are recorded before formal launch.

The logical mask implements weighted attention. KVs remain allocated: no byte,
latency or throughput reduction is claimed. Total-context retention varies with
prompt length; all paired comparisons have identical logical budgets.

## Primary measurements

For each Q policy, generate its own continuation and replay dense P on those
same tokens. Sum exact full categorical `KL(Q_t(.|Y_<t)||P_t(.|Y_<t))`; averaging
Q paths estimates sequence KL(Q||P). Average phases/replicates within prompt,
then give each of the twelve prompts equal weight. This is a mean conditional
on compressor schedules, not KL of a hidden-schedule mixture.

At the first query affected by each edit, measure the direct pre-o_proj attention
displacement with versus without **only that latest edit**, holding that layer's
query and accumulated Q cache fixed. Sum squared components over heads and
head dimensions; average over layers/events. The local metric matches pooled
magnitude, not vectors, layer profiles, output sensitivity or history laws.

Save per-step conditional KL, event/layer local errors, prompt lengths, selected
schedules/seeds, sampled token IDs, EOS IDs/first EOS position and KL milestones.
Generation continues after EOS to the fixed horizon. This defines a truncated
raw-model token law, not a normal stop-on-EOS assistant completion.

## Analysis and failure interpretation

The primary comparison is Block minus Alternating KL, per prompt and as an
equal-prompt mean. Report both differences and ratios, all per-prompt results,
all three policy aggregates and local-error ratios. A local max/min ratio <=1.25
tags comparability. Count prompts also showing Block/Alternating KL ratio >=2
as a descriptive effect-size screen, not a theorem or mandatory success gate.
Show absolute KL values because ratios near zero are unstable.

Keep every non-comparable and negative prompt. No prompts, seeds, horizons,
checkpoints or thresholds may be selected to rescue the contrast. Random is a
descriptive control; it need not fall between the deterministic policies.
Prompt bootstrap intervals describe variation over these fixed templates within
one model; nested paths are not independent models or task-population samples.
Report Monte Carlo path dispersion separately where available, without counting
token positions as independent samples. Flag early EOS and per-policy incidence;
do not infer natural completion quality from post-EOS divergence.

If local comparability fails, the trial does not isolate magnitude from other
selection changes. If no material contrast occurs, report the absence on this
checkpoint and prompt set. A positive result alone does not identify signed
error cancellation. Common-history replay is a possible subsequent bounded
mechanism test, not silently added to this launch. No second checkpoint, new
Fisher computation, training or parameter sweep is included.

## Minimal correctness checks and dispatch

1. CPU schedule/weighted-mask/KL identity and complete-summary tests.
2. Random tiny Qwen2: native vs zero-mask decode, both physical-vs-logical
   orientations with original positions, hook output invariance and probe count.
3. One-prompt T=8, one-path-per-phase 7B smoke (six rows), using the same code.
4. If these pass, freeze source/config hashes and launch all 144 formal rows.

Use a new output directory for every attempt. Analysis refuses incomplete grids
and requires a completed manifest. Preserve failed attempts and logs. Wall-time
cap is 10,800 seconds; runtime estimates use only smoke timing, not its outcomes.
Record the host, GPU, PID, command, UTC time and output in a private dispatch
receipt. A verified persistent launch completes this authorized dispatch task;
there is no recurring monitor.

Canonical configuration: `real_llm_temporal_v1/config.json`.
Runner: `real_llm_temporal_v1/run_qwen_probe.py`.
Original package: `../L01_RealLLM_Temporal_Order_Pilot.zip` (preserved unchanged).
