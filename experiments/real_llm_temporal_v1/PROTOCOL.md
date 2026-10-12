# Original supplied pilot protocol (superseded for execution)

The canonical v1 protocol is [PROTOCOL_REAL_LLM_TEMPORAL_V1.md](../PROTOCOL_REAL_LLM_TEMPORAL_V1.md).
The text below is retained as the supplied package's design provenance. The v1
protocol/config and frozen source manifest control the authorized experiment.

## Scientific question

Does the L01 Round 12 observation — almost matched **instantaneous local attention error** but substantially different **full Q-sampled sequence KL** under temporally different cache-selection schedules — appear in a pretrained/instruction-tuned real LLM outside the histogram toy?

**No affirmative real-LLM result is claimed in this package.** The packaging environment does not have a CUDA GPU, weights or `transformers`. It contains a runnable Qwen2-specific integration harness and a completed CPU algebraic self-test only.

## Pre-registered main contrast

Target `Qwen/Qwen2.5-7B-Instruct`, BF16, full-vocabulary softmax at temperature 1, no top-k/top-p, eager attention; pin `transformers==4.57.1`. Twelve public synthetic prompts span reasoning, narrative, and instruction following, with Chinese and English represented. Each Q policy generates **T=130** tokens. The source prompt is exact, and each cache appends its own generated-suffix K/V. For each schedule we compare with dense P on the **same Q-sampled token history**.

At 64 post-pair events, each two consecutive generated KV records is represented by one of their original records at **weight 2**, leaving all other records at unit weight. Plus selects the **earlier** record, minus the **later**; orientation does not depend on token ID. Every schedule includes its sign-inverted phase. For each prompt, two independent sampling paths per phase. All three policies are evaluated at the same token horizon, with exactly 64 events:

1. Balanced Block: + repeated 32 events, then − repeated 32 events (and antipode).
2. Balanced Alternating: +−+−… (and antipode).
3. Balanced Random: a fixed-seed random permutation of 32 plus and 32 minus (and antipode).

**There is no claim of byte or throughput reduction.** The 7B harness uses additive 4D masks to emulate the attention operation exactly: a survivor receives `+log(2)`, a discarded atom `-inf`. Both K/V remain allocated for measurement. Mathematical equivalence of the weighted attention operation is checked in the CPU self-test. **Tiny HF physical-vs-logical parity must pass** on the user's real Transformers installation before 7B measurement.

## Autoregressive risk and controls

For each fixed prompt and compressor schedule, sample the Q continuation from the **full softmax** and replay P on these same Q tokens. Record:

`KL(Q_1:T || P_1:T) = E_{Y~Q} sum_t KL(Q_t(.|Y_<t) || P_t(.|Y_<t))`.

The trajectory Monte Carlo estimator sums exactly evaluated categorical conditional KLs. The result is **mean fixed-schedule KL**, NOT KL after marginalizing the schedule randomness. Common inverse-CDF uniform draws couple the two policies and their antipodal phases, but do not force their histories to be identical.

At the first token affected by each edit, the `Qwen2` eager-attention hook computes direct pre-output-projection attention displacement, comparing **same query and same Q cache** under the real edit vs. undoing only that newest edit. Earlier edits remain intact. Average its squared full-head output norm over layers/events. This is a direct **local attention error**; it is not a task metric or cumulative KL. This instrumentation does not change the Q forward result, and is intentionally descriptive.

The protocol additionally records event-level errors and full-sequence KL milestones (16,32,64,96,130), policy outputs, EOS incidence, prompt identity/category, model config and library versions. A row cannot be selected based on outcome. Use full-cache identity and custom-mask parity tests.

## Pilot vs independent evidence

- **Smoke**: 1 prompt, `T=8`, `paths_per_phase=1`. Run `--self-test` and `--hf-smoke-tiny` first.
- **Fixed 7B pilot**: all 12 prompts, T=130, 2 paths/phase, all schedules; NO prompt-based post hoc selection. The natural-language prompts in `prompts.json` are fully frozen by SHA.
- **Replication**: if available, run the same package without tuning on a second Qwen2-family trained checkpoint (e.g. Qwen2.5-3B-Instruct). Replication across a different model family requires separately implementing and verifying attention instrumentation (not implemented here). Twelve prompts are not twelve independent model checkpoints.

Primary descriptive comparison is **Block vs Alternating**: mean full KL and mean local squared-error per prompt (averaging phases and paths). Each prompt receives equal weight. Report *both* the KL difference and ratio; ratios are unstable near zero. A prompt is 'locally comparable' only when the larger/smaller mean local squared error ratio <=1.25. **All non-comparable prompts must remain in the primary table**, separately tagged. Also report balanced-random controls and EOS frequency. Paired prompt-cluster bootstrap gives only a *within-fixed-model prompt-variation* interval, not uncertainty across pretrained models.

A more convincing study would show a material KL gap on many independent prompts **while** local distortion is comparable, with the random-permutation control intermediate or otherwise informative, plus evidence that the effect survives another checkpoint. There is **no guarantee the effect exists** on a general LLM. If local error comparability fails, report the failure without claiming a covariance-isolated effect. If early EOS is common, stop using this fixed-length generation setup as evidence about natural completion quality.

## Mechanism audit if the primary gap exists

1. **Accumulation vs terminal state.** Contrast KL at intermediate steps with final KL, and inspect event-level mean local error and history variation. KL terms cannot be retroactively canceled, even if a later cache becomes similar.
2. **Temporal dependence vs magnitude.** Use the balanced-random schedule. Compute time trajectories of cumulative selected KV feature discrepancies on fixed Q paths (future extension; not in v0 harness). Avoid claiming that equal mean MSE implies equality of full error distributions.
3. **Closed-loop shift.** On a modest frozen subset, replay P-generated text through every compressed schedule, giving a P-history conditional-KL diagnostic; compare to the Q-sampled risk. The P-history quantity is not KL(Q||P).
4. **Failure boundary.** Compare with Round 14B and do not assume finite full-strength compression is well approximated by zero-strength Fisher. A 64-event full-network Fisher on a 7B model is expensive and is NOT a precondition for the primary phenomenon.

## Novelty threats and claim limits

BalanceKV (NeurIPS 2025) already demonstrates geometry-aware dependent KV selection. ACL 2026 *Pitfalls of KV Cache Compression* studies real LLM instruction-order/eviction bias. ICML 2026 *Semantic Integrity Matters* studies task-dependent CoT vulnerability. An LLM gap **alone** is not original unless local distortion/budget confounding is handled and temporal interaction is isolated more sharply. A clean negative result is equally important. Do not describe this as an online optimal compressor, efficiency breakthrough, all-LLM generalization or rigorous certification.

## Commands on an RTX 5090 workstation

```bash
python -m pip install 'transformers==4.57.1' 'safetensors' 'accelerate'
python run_qwen_probe.py --self-test
python run_qwen_probe.py --hf-smoke-tiny
python run_qwen_probe.py --model-path /LOCAL/Qwen2.5-7B-Instruct --limit-prompts 1 --horizon 8 --paths-per-phase 1 --out-dir runs/smoke_new
python run_qwen_probe.py --model-path /LOCAL/Qwen2.5-7B-Instruct --out-dir runs/frozen_pilot_new
```

The script refuses to overwrite an existing run directory. If a run fails after creating the folder, use a new folder for any retry and retain the failed log. Supply the real local model path; no particular Windows/AutoDL server path is assumed. If the HuggingFace tiny parity test fails, **do not** interpret the 7B results; correct the implementation before starting.

## Current verification status

- **Passed here**: Torch CPU schedule/antiphase check, mathematical `(2,0)` weighted softmax/mask equality across randomized fixtures, full-softmax KL identity.
- **Not run here**: `transformers` 4D mask integration, tiny Qwen2 forward parity, 7B prefill/decode, instrumented attention hooks, any actual 7B KL. The current container has no CUDA GPU and no installed Transformers library. These are hard blockers to reporting an empirical result in this response.
