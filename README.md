# L01: causal output-relevant KV error

## Latest: 7B pilot complete; large ordering gap not observed

L01 asks which KV perturbations explain autoregressive output error. The new
Qwen2.5-7B-Instruct pilot completed all 144 trajectories on twelve fixed prompts.
Balanced Block/Alternating mean sequence KL is 181.3447/185.2281 nats. Eleven
prompts have comparable pooled local attention-error magnitudes, but none has
a twofold KL gap in either direction. The earlier histogram model's 11--12x
contrast does not reproduce here. This single-checkpoint pilot does not establish
equivalence, task quality, a compression certificate or a systems benefit.

1. [RESULTS.md](RESULTS.md): the 7B table and claim limits, followed by earlier results.
2. [GPT_HANDOFF.md](GPT_HANDOFF.md), then the new section of [GPT_CONTEXT.md](GPT_CONTEXT.md).
3. [7B interpretation](runs/real_llm_temporal_v1_20261012T123114Z/INTERPRETATION.md) and [small summary](runs/real_llm_temporal_v1_20261012T123114Z/evaluation/summary.json): all-prompt comparisons and uncertainty.
4. [Frozen protocol](experiments/PROTOCOL_REAL_LLM_TEMPORAL_V1.md), [config](experiments/real_llm_temporal_v1/config.json), [runner](experiments/real_llm_temporal_v1/run_qwen_probe.py) and [source identities](runs/real_llm_temporal_v1_20261012T123114Z/FROZEN.json).

The 8 MB trajectory JSONL is secondary evidence. Verify published records with
Python's standard library, without loading weights or rewriting frozen files:

```text
python -B scripts/verify_real_llm_temporal_v1_release.py --input-dir runs/real_llm_temporal_v1_20261012T123114Z
```

For a fresh GPU reproduction, use the cached model snapshot named in the protocol
and the separate [7B dependencies](experiments/requirements_real_llm_temporal_v1.txt);
the root requirements belong to earlier experiments. From this repository root:

```text
python -B -m unittest discover -s experiments/real_llm_temporal_v1 -p test_cpu.py -v
python -B experiments/real_llm_temporal_v1/run_qwen_probe.py --model-path MODEL_SNAPSHOT --out-dir runs/NEW_REAL_LLM_RUN
```

The runner uses local-only model loading and refuses an existing output directory.
Original package descriptions and the dispatch snapshot are preserved historical
records; their pending-result language is superseded by the completed evidence.
The input ZIP remains a local provenance archive, identified by SHA-256 in the
source manifest; it is not needed to read or reproduce the published source.

## Historical: Round 14B final run closed without establishing transfer

L01 asks which KV perturbations explain autoregressive output error. The final
suffix-write recall test makes compressed values necessary for retrieval and
adds an actual Round 7 one-shot GS/uniform control. Execution completed, but
only 2/5 seeds meet prerequisites and 0/5 meet all frozen gates. Full Fisher
severely underestimates finite-endpoint KL even in both qualified models.
Round 13's narrow histogram result stands; broader trained-model transfer is
not established. No failed seed was removed and no rescue round follows.

1. [RESULTS.md](RESULTS.md): Round 14B endpoint, one-shot control and historical results.
2. [GPT_HANDOFF.md](GPT_HANDOFF.md), then [GPT_CONTEXT.md](GPT_CONTEXT.md): incremental review and exact source map.
3. [Interpretation](runs/round14b_suffix_write_v1_20261012T102704Z/INTERPRETATION.md), [report](runs/round14b_suffix_write_v1_20261012T102704Z/evaluation/REPORT.md), [summary](runs/round14b_suffix_write_v1_20261012T102704Z/evaluation/summary.json), and [closure](runs/round14b_suffix_write_v1_20261012T102704Z/CLOSURE.md).
4. [Frozen protocol](experiments/PROTOCOL_ROUND14B_SUFFIX_WRITE_V1.md), [configuration](experiments/configs/round14b_suffix_write_v1.json), and [source hash manifest](experiments/FROZEN_ROUND14B_SUFFIX_WRITE_V1.json).

Read summaries before the secondary JSONL/NPZ evidence. Audit saved records
without checkpoints, training, generation, or JVP replay:

```text
python -B scripts/verify_round14b_release.py --input-dir runs/round14b_suffix_write_v1_20261012T102704Z
```

The protocol includes fresh-run reproduction commands; dependencies are in
[requirements.txt](requirements.txt). Checkpoint bytes and private receipts
are excluded. Full/diagonal Fisher is a retrospective Q-history diagnostic;
theta=.25 uses extra atoms and is not a physical memory-saving endpoint.
The one-shot and continuous arms have separate budgets and do not isolate
compression timing at a matched total budget. Nonzero empirical KL does not
refute the separately conditional Round 7 bound.

The [novelty audit](NOVELTY_AUDIT_2026-10-12.md) and
[pre-run protocol review](ROUND14B_PROTOCOL_REVIEW_2026-10-12.md) are historical
decision notes, not new theorems or publication-readiness conclusions.

## Historical: Round 14 associative-recall transfer is not established

L01 asks which aspects of KV perturbations explain autoregressive output error.
Round 13 accurately explained a finite histogram-model contrast using full
cross-event Fisher geometry. Round 14 tests one new eight-key generative
associative-recall task: five newly trained two-layer models, two heldout
tables, and the same T=129 suffix-thinning comparison. All runs completed,
but only 4/5 models learned retrieval well enough and both frozen Fisher
transfer screens fail. The earlier histogram result stands; quantitative
transfer beyond it is not established. Failed models remain in the evidence.

1. [RESULTS.md](RESULTS.md): Round 14 qualification and negative transfer endpoint, then historical results.
2. [GPT_HANDOFF.md](GPT_HANDOFF.md), then [GPT_CONTEXT.md](GPT_CONTEXT.md): incremental changes, exact quantities and source map.
3. [Round 14 interpretation](runs/round14_recall_transfer_review_v1_20261012T085656Z/INTERPRETATION.md), [report](runs/round14_recall_transfer_v1_20261012T083530Z/evaluation/analysis_round14/REPORT.md) and [summary](runs/round14_recall_transfer_v1_20261012T083530Z/evaluation/analysis_round14/summary.json).
4. [Frozen protocol](experiments/PROTOCOL_ROUND14_RECALL_TRANSFER_V1.md) and [configuration](experiments/configs/round14_recall_transfer_v1.json): task adaptation, training, qualification, estimands and reproduction.

Read summaries before raw NPZ and prefix/event CSV evidence. Check the
published records without checkpoints:

```text
python -B scripts/verify_round14_release.py --input-dir runs/round14_recall_transfer_v1_20261012T083530Z
```

Round 14 can be trained from scratch with the protocol's fixed recipe and
[requirements.txt](requirements.txt); checkpoints are excluded from GitHub.
Use fresh training/evaluation directories. Generation KL and Fisher share
evaluated Q histories; Fisher is not heldout forecasting or a certificate.
Only eligible generated suffix pairs are thinned; no whole-cache half-budget
or inference speed claim is made.

## Historical: Round 13 cross-event explanation at the primary horizon

Round 12 found that Balanced Block versus Alternating thinning has similar
terminal token bias but an 11.17x full-sequence KL contrast; the whole prefix
bias trajectory tracks the contrast. Round 13 now tests exact full-unroll
Fisher geometry at the same T=129 and KL(Q||P) direction, reusing the five
original two-layer models and saved histories without training. The selected
history KL ratio is 12.19x; full Fisher gives 12.24x, while diagonal-only gives
0.98x. Both frozen screens pass all five seeds for each policy.

This is a finite same-history explanation, not a heldout predictor or
certificate. The quadratic uses conditional P-logit geometry averaged over
saved Q histories. Only generated suffix pairs are thinned; whole-cache
half-budget and inference speed were not measured. Earlier certificate
failures remain unchanged.

1. [RESULTS.md](RESULTS.md): Round 13 and Round 12 endpoint tables and limits.
2. [GPT_HANDOFF.md](GPT_HANDOFF.md), then [GPT_CONTEXT.md](GPT_CONTEXT.md): incremental review and exact estimands/source symbols.
3. [Round 13 interpretation](runs/round13_matched_fisher_review_v1_20261012T071747Z/INTERPRETATION.md), [report](runs/round13_matched_fisher_v1_20261012T062406Z/REPORT.md) and [summary](runs/round13_matched_fisher_v1_20261012T062406Z/summary.json).
4. [Round 13 protocol](experiments/PROTOCOL_ROUND13_MATCHED_FISHER_V1.md) and [configuration](experiments/configs/round13_matched_fisher_v1.json); [Round 12 interpretation](runs/round12_balanced_order_review_v1_20261012T060523Z/INTERPRETATION.md) supplies the actual bias trajectories.

Read compact summaries before the secondary raw NPZ/CSV files. Check the
recorded Round 13 endpoint without model weights:

```text
python -B scripts/verify_round13_release.py --input-dir runs/round13_matched_fisher_v1_20261012T062406Z --input-run runs/round12_balanced_order_v1_20261012T052457Z
```

Reproduce with the original external checkpoints and a fresh output directory:

```text
python -B experiments/round13_matched_fisher.py --config experiments/configs/round13_matched_fisher_v1.json --checkpoint-root PATH_TO_ORIGINAL_MODELS --input-run runs/round12_balanced_order_v1_20261012T052457Z --run-dir runs/NEW_ROUND13_RUN
```

Dependencies are in [requirements.txt](requirements.txt). Exact replay needs the
original checkpoint bytes; replacement training is a new realization. The
runner writes analysis automatically; reanalysis must use a fresh copy of
`config.json` and `metrics.csv`, not the frozen completed directory.

## Historical: Round 11 streaming covariance

The completed Round 11 study tests how temporal correlation between equally budgeted suffix-thinning decisions changes sequence risk in the five original two-layer histogram decoders. It is a finite frozen-checkpoint mechanism result: no training or certification was performed.

Endpoint A is `ROUND10_ORDER_TRANSFER_WITH_LOCAL_COMPARABILITY_SCREEN`: at T=128, mean fixed-schedule KL(Q||P) orders coherent > IID > alternating in all five model seeds. The max/min ratio of mean direct local-attention squared error is 1.00554–1.01176 across seeds, below the 1.25 screen. The paired coherent-minus-alternating difference is 2.62975, with a descriptive 95% training-seed bootstrap interval [2.57981, 2.68374].

Endpoint B is `FINITE_FISHER_EXPLANATION_SUPPORTED` at amplitude 0.25 and T=32. On fixed P histories, the full reference-path Fisher prediction explains finite reverse sequence KL(P||Q) in all five seeds for coherent and alternating schedules: relative errors range from 0.0000359 to 0.000598, while diagonal-only errors range from 0.373 to 1.125. Both endpoints passed their predeclared screens; the run contains 960/960 primary rows and 60/60 Fisher rows.

The primary metric averages fixed-schedule Q-to-P sequence KLs; it is not KL of a mixture over hidden schedules. Endpoint B uses the opposite direction, P-to-Q, on fixed P histories. Only eligible generated suffix pairs are thinned; the old prefix and START remain intact, so the result does not show that the full cache is halved. No byte, memory, or speed measurement was made.

Use the [corrected Fisher figure](runs/round11_streaming_covariance_v1_20261012T042125Z/plot_correction/fisher_predictions.png): the frozen original plot applied amplitude squared twice. Numerical CSV results and screens are unchanged; the original plot is retained with a [correction record](runs/round11_streaming_covariance_v1_20261012T042125Z/plot_correction/README.md).

### Round 11 reading route

1. [RESULTS.md](RESULTS.md), then the [run interpretation](runs/round11_streaming_covariance_v1_20261012T042125Z/INTERPRETATION.md): canonical endpoint tables, screens, and caveats.
2. [GPT_HANDOFF.md](GPT_HANDOFF.md), then [GPT_CONTEXT.md](GPT_CONTEXT.md): incremental review, formal status, estimands, source map, and review boundaries.
3. [Frozen protocol](experiments/PROTOCOL_ROUND11_STREAMING_COVARIANCE_V1.md) and [configuration](experiments/configs/round11_streaming_covariance_v1.json).
4. [Run summary](runs/round11_streaming_covariance_v1_20261012T042125Z/summary.json) and [run configuration](runs/round11_streaming_covariance_v1_20261012T042125Z/config.json) bind the aggregate and evaluated grid. The large per-event CSV is secondary; begin with RESULTS.md and summary.json.

The public endpoint is checked by the read-only release verifier:

```text
python -B scripts/verify_round11_release.py --input-dir runs/round11_streaming_covariance_v1_20261012T042125Z
```

To reproduce, use the original external checkpoints and a fresh run directory:

```text
python -B experiments/round11_streaming_covariance.py --config experiments/configs/round11_streaming_covariance_v1.json --checkpoint-root PATH_TO_ORIGINAL_MODELS --run-dir runs/NEW_ROUND11_RUN
```

The runner generates the analysis automatically after scientific completion.
For a separate analysis of the frozen run, copy `primary_metrics.csv`,
`fisher_metrics.csv`, `execution_summary.json`, `provenance.json`,
`checkpoints.json`, `prefixes.json`, `config.json`, and `source/` into a new directory, then pass that
directory as `--run-dir` with the frozen run's `config.json`. The analyzer
writes its derived summary and figures there; keep this output away from the
frozen completed run.

## Historical: Round 9 local-bound certificate campaign

Round 9 asked whether local analytic bounds certify substantial prefix-KV compression on the original five two-layer histogram models. All 330 conditions completed: at half cache (n=64,m=32), no method passes the frozen KL<=0.1 feasibility gate. Agglomerative actual KL is tiny, but its mean upper bound is 8.4763; near-full n=64,m=63 succeeds. [RESULTS.md](RESULTS.md) distinguishes these negative and positive findings. Earlier Round 7, Round 6 and Experiment 0 results are preserved.

Two subsequent frozen one-case CPU follow-ups find an accurate first-order
JVP, but a scalar second-order remainder enclosure still gives sequence KL
upper 30.9056 after intersection with the original bound. The <=0.1 target
remains unmet; this does not establish that all directional methods fail.

A completed finite-task result does not establish a general trained-Transformer theorem, a uniform generation guarantee, minimax cache-space lower bound, sequence-TV certificate, or practical inference or memory savings. Treat task-quality failures as part of the reported seed results; do not silently remove them.

## Round 9 historical reading route

1. [RESULTS.md](RESULTS.md): frozen Round 9 verdict, half-cache table and near-full control.
   The new [one-case diagnostic](runs/round9_directional_diagnostic_v1_20261011T183000Z/INTERPRETATION.md)
   separates propagation/readout slack and checks the full-network JVP; it does
   not change the failed half-cache certificate gate.
   The subsequent [remainder attempt](runs/round9_directional_remainder_v1_20261012T024528Z/INTERPRETATION.md)
   also fails: preserving the first-order direction does not by itself keep
   a scalar second-order enclosure small.
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
2. [GPT_CONTEXT.md](GPT_CONTEXT.md): preserved Round 7 context; the current [GPT_HANDOFF.md](GPT_HANDOFF.md) covers the Round 11 update.
3. [GPT_CONTEXT.md](GPT_CONTEXT.md), [frozen Round 7 protocol](experiments/PROTOCOL_ROUND7_THINNING_V1.md), and [configuration](experiments/configs/round7_thinning_v1.json): estimands, source map and controls.
4. [THEORY_STATUS.md](THEORY_STATUS.md), then the [archived Round 7 proof draft](theory/archive/round7/THEOREM_AND_PROOF.md): conditional mathematics and restricted lower-bound class.

Current endpoint status and numbers belong in RESULTS.md; this reading guide does not duplicate them.

## Historical: Round 7 verification and reproduction

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

## Historical: Round 6 reproduction

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
