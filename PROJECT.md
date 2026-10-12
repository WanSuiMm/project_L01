# L01: output-relevant causal KV error

Status: **ROUND13_COMPLETE / MATCHED_HORIZON_FISHER_SCREEN_PASS_5_OF_5 / CROSS_EVENT_EXPLANATION_SUPPORTED**;
previous **ROUND12_COMPLETE / BALANCED_ORDER_SCREEN_PASS_5_OF_5 / PREFIX_BIAS_TRAJECTORY_SUPPORTED**;
previous **ROUND11_COMPLETE / TEMPORAL_ORDER_SCREEN_PASS_5_OF_5 / LOCAL_FISHER_SCREEN_PASS_5_OF_5**;
previous **ROUND9_REMAINDER_FEASIBILITY_FAIL / FIRST_ORDER_DIAGNOSTIC_STILL_PASS / HALF_CACHE_UNCERTIFIED**;
previous **ROUND9_DIAGNOSTIC_COMPLETE / ONE_CASE_FIRST_ORDER_SCREEN_PASS / HALF_CACHE_STILL_UNCERTIFIED**;
previous **ROUND9_COMPLETE / HALF_CACHE_FEASIBILITY_FAIL / NEAR_FULL_CLUSTER_ONLY**;
previous **ROUND7_EXECUTION_COMPLETE / GS_EMPIRICAL_ADVANTAGE / GLOBAL_CERTIFICATE_VACUOUS**;
previous **ROUND6_EXECUTION_COMPLETE / ANALYTIC_CONSISTENT /
HISTOGRAM_QUALIFIED / RECALL_TRAINING_UNRELIABLE / GLOBAL_BOUND_UNINFORMATIVE**,
2026-10-12.

Round 13 completed on 2026-10-12 at 06:44:26 UTC in 634.65 scientific seconds
(635.60 supervisor seconds), CPU two threads, float64. Its identity is
`round13_matched_fisher_v1_20261012T062406Z`; all 640 metric rows, 40 trace files
and 80 event Grams were retrieved. Terminal receipts remain private.

The frozen [matched-horizon Fisher protocol](experiments/PROTOCOL_ROUND13_MATCHED_FISHER_V1.md)
reuses the same five two-layer models, four prefixes, T=129, and both antipodal
phases for Balanced Block and Alternating. It uses the first 16 saved Round 12
Q-law histories per phase without training or generating new tokens. Exact
full-unroll event JVPs compare all cross-event Fisher terms against a
diagonal-only approximation, alongside endpoint conditional KL(Q||P) chain
sums on those same histories. The Fisher quantity is evaluated at the exact
model and averaged over saved Q histories; it is not the sequence Fisher
expectation under P or a heldout prediction. Replay at amplitude 0.25 is a
conditional diagnostic rather than sequence KL under that amplitude's law.

Two targeted tests, original-model local/server smoke checks, and a full-length
single-condition timing calibration pass. Calibration took 17.07 seconds per
condition, so the frozen runtime rule retained 16 histories per phase. Both
predeclared endpoint screens pass in all five seeds for each policy. Actual
Block/Alternating KL is 1.216538/0.0997674; full Fisher is 1.220090/0.0997077,
and diagonal Fisher is 0.178589/0.181480. Their ratios are 12.1937, 12.2367
and 0.984066. Mean per-seed full relative errors are 0.322% and 0.171%.
This supports a finite same-history cross-event explanation; it is not a
P-law sequence-Fisher theorem, heldout predictor or useful certificate.
Read [RESULTS.md](RESULTS.md), then the
[interpretation](runs/round13_matched_fisher_review_v1_20261012T071747Z/INTERPRETATION.md).

Round 13 commands from the repository root (use a new run directory):

```text
python -B -m unittest discover -s experiments -p test_round13_fisher.py -v
python -B scripts/verify_round13_release.py --input-dir runs/round13_matched_fisher_v1_20261012T062406Z --input-run runs/round12_balanced_order_v1_20261012T052457Z
python -B experiments/round13_matched_fisher.py --config experiments/configs/round13_matched_fisher_v1.json --checkpoint-root PATH_TO_ORIGINAL_MODELS --input-run runs/round12_balanced_order_v1_20261012T052457Z --run-dir runs/NEW_ROUND13_RUN
```

Round 12 completed on 2026-10-12 at 05:31:21 UTC in 86.37 scientific seconds
using CPU two threads and float64. Its identity is
`round12_balanced_order_v1_20261012T052457Z`; frozen outputs and private terminal
receipts were retrieved. No ongoing monitor was created. Both predeclared
descriptive screens pass in all five training seeds.

The frozen [Balanced Block vs Alternating protocol](experiments/PROTOCOL_ROUND12_BALANCED_ORDER_V1.md)
reuses all five original two-layer models and the same four n=64 Round 11
prefixes, without training. T=129 gives exactly 64 generated-suffix pair
thinning events: 32 positive then 32 negative versus alternating signs,
with both antipodal phases and 64 Q-law paths per phase. Every prediction
prefix retains conditional and cumulative KL(Q||P), actual token-histogram
discrepancy, direct cumulative KV perturbations, and actual weighted cache
moment discrepancy against full P replay of the same Q history. Joint
path trajectories and layer diagnostics are retained. Sign balance does
not imply token/KV bias matching; earlier certificate failures are unchanged.

Four targeted software tests pass, including enumerable-tree sequence KL
versus its conditional chain sum. Original seed-11 smoke runs pass locally
and on the server; the four fixed prefixes were checked identical to Round 11.
These checks validate implementation, not scientific hypotheses. Formal
coverage is 10,320 prefix rows, 10,240 event/layer/phase rows, 5,120 terminal
path rows and 40 joint trajectory files, all retrieved. Mean full-sequence
KL(Q||P) is 1.201930 for Block versus 0.107638 for Alternating (11.17x).
Terminal token-bias squared is only 1.04x, while integrated prefix token-bias
squared is 11.45x and cache-moment bias energy is 6.70x. This supports the
actual bias trajectory as an empirical explanation, not sole-cause identification
or a heldout quantitative risk law. Read the
[result interpretation](runs/round12_balanced_order_review_v1_20261012T060523Z/INTERPRETATION.md),
then the [original report](runs/round12_balanced_order_v1_20261012T052457Z/REPORT.md)
and [joint recorded-evidence check](runs/round12_balanced_order_review_v1_20261012T060523Z/joint_bias_kl_review.json).
The Round 13 full-unroll Fisher comparison now completes the finite bridge at
the same horizon and KL direction as the primary risk, as described above.

Round 12 commands from the repository root (use a new run directory):

```text
python -B -m unittest discover -s experiments -p test_round12_balanced_order.py -v
python -B experiments/round12_balanced_order.py --config experiments/configs/round12_balanced_order_v1.json --checkpoint-root PATH_TO_ORIGINAL_MODELS --run-dir runs/NEW_ROUND12_RUN
python -B scripts/inspect_round12_result.py --run-dir runs/round12_balanced_order_v1_20261012T052457Z --output-dir runs/NEW_ROUND12_INSPECTION
```

Round 11 completed on 2026-10-12 at 04:28:23 UTC in 117.40 seconds including
analysis, CPU two threads. Its frozen
[mechanism protocol](experiments/PROTOCOL_ROUND11_STREAMING_COVARIANCE_V1.md)
reuses the five original width-64, two-layer Round 6 histogram checkpoints
without training. It compares coherent, IID and alternating pair-thinning
policies over four prefixes at T=16,32,64,128. The old prefix remains exact;
only eligible newly generated suffix pairs are halved. This is not a
whole-cache half-budget experiment. A separate fixed-reference full-unroll
Fisher probe tests whether cross-event terms improve finite reverse-KL
predictions. Actual Q-to-P sequence risk and this P-to-Q diagnostic are
reported separately.

Four targeted core tests pass. Original seed-11 smoke runs pass locally and
on the server, including original-model, physical-endpoint and JVP parity;
the analysis entry point also runs on both environments. These checks are
software validation only. All 960 primary rows, 60 Fisher rows and 7,560
event rows completed and were retrieved. At T=128 mean KL(Q||P) is 2.730185
for coherent, 0.175011 for IID and 0.100431 for alternating; all five models
pass the ordering and local-error comparability screens. The T=32 reverse-KL
Fisher screen passes all five models, but is a distinct local diagnostic,
not a direct explanation of the whole long-horizon primary gap. See
[RESULTS.md](RESULTS.md) and the
[run interpretation](runs/round11_streaming_covariance_v1_20261012T042125Z/INTERPRETATION.md).
Source/configuration snapshots and retrieved outputs are in that run
directory; operational receipts remain private. Earlier certification
failures are unchanged. No ongoing monitor was created.

Publication review found double amplitude scaling in the original Fisher
plot only. Current analyzer code and separate `plot_correction/` figures fix
the plot; original scientific metrics and frozen artifacts are unchanged.

Round 11 commands from the repository root (always use a new run directory):

```text
python -B -m unittest discover -s experiments -p test_round11_streaming_core.py -v
python -B scripts/verify_round11_release.py --input-dir runs/round11_streaming_covariance_v1_20261012T042125Z
python -B experiments/round11_streaming_covariance.py --config experiments/configs/round11_streaming_covariance_v1.json --checkpoint-root PATH_TO_ORIGINAL_MODELS --run-dir runs/NEW_ROUND11_RUN
```

The runner generates the analysis automatically. For reanalysis, copy compact
inputs and source to a fresh directory as described in README.md.

The next frozen attempt kept exact signed first-order directions and bounded
second-derivative norms over theta in [0,1]. All 40 histories completed on the
same case. Sequence upper improves from 36.6231 to 30.9056 after intersection
with old conditional bounds; target<=0.1 fails. Raw directional upper is
3.43908e17. Empty-history upper is already 3.32726, so rare histories do not
alone explain failure. Four software tests and retained-record verification
pass. Read the [remainder interpretation](runs/round9_directional_remainder_v1_20261012T024528Z/INTERPRETATION.md)
and [method note](theory/ROUND9_DIRECTIONAL_REMAINDER_V1.md). CPU calculation took
1.36 seconds without training. These are float64 analytic-bound evaluations,
not outward-rounded certificates. Commands from the repository root:

```text
python -B -m unittest discover -s experiments -p test_round9_directional_remainder.py -v
python -B scripts/verify_round9_directional_remainder.py --input-dir runs/round9_directional_remainder_v1_20261012T024528Z
python -B experiments/round9_directional_remainder.py --config experiments/configs/round9_directional_remainder_v1.json --checkpoint-root PATH_TO_ORIGINAL_MODELS --run-dir runs/NEW_REMAINDER_RUN
```

A frozen follow-up used seed 11 and the first n=64 prefix, positive cluster
m=32, T=4, and all 40 histories. The full-unroll JVP approximates the endpoint
centered-logit change with median 1.85% and maximum 6.37% scaled vector remainder.
The hidden radius overestimates actual error by median 166.4x (range 114.4–295.1x),
versus median 5.75x additional centered-head norm slack. This is one-case oracle
diagnosis, not a replacement certificate; the Round 9 gate remains failed.
Read the [diagnostic interpretation](runs/round9_directional_diagnostic_v1_20261011T183000Z/INTERPRETATION.md),
then its generated report and verification. Four targeted software tests pass;
the scientific CPU calculation took 3.48 seconds without retraining. Canonical
diagnostic commands from the repository root are:

```text
python -B -m unittest discover -s experiments -p test_round9_directional_diagnostic.py -v
python -B scripts/verify_round9_directional_diagnostic.py --input-dir runs/round9_directional_diagnostic_v1_20261011T183000Z
python -B experiments/round9_directional_diagnostic.py --config experiments/configs/round9_directional_diagnostic_v1.json --checkpoint-root PATH_TO_ORIGINAL_MODELS --run-dir runs/NEW_DIAGNOSTIC
```

Round 9 launched reference-only finite-tree local-bound evaluation on the five
original qualified two-layer histogram checkpoints. The frozen
[protocol](experiments/PROTOCOL_ROUND9_LOCAL_CERT_V1.md) specifies 330 conditions:
three fixed prefixes at n=16 and 64, T=4 over the full three-token tree,
near-full and 25%/50%/75% old-prefix record reductions, with positive cluster,
uniform subset and eligible GS-Thin summaries. Eight local software tests pass.
The formal CPU run completed all 330 conditions on 2026-10-12 at 01:43:41
Asia/Shanghai in 67.78 seconds. All five checkpoint hashes and recorded quality
results matched; no training occurred. At n=64,m=32 all methods have 0/5 passing
models and 0/15 prefixes with upper KL<=0.1. Agglomerative mean upper KL is
8.4763 versus actual KL 5.4366e-6; its bound improves on the trivial cap but
does not meet the frozen high-fidelity gate. Only near-full agglomerative
n=64,m=63 has 15/15 upper KL<=0.1. Read [RESULTS.md](RESULTS.md), then the
[completed run status](runs/round9_local_cert_v1_20261011T045942Z/RUN_STATUS.md).
The compact snapshot retains all 330 condition metrics, with independently
checked coverage, seed/group arithmetic and gates. Earlier failed connections
and dispatch records remain private. Round 6/7 results are unchanged. These
are float64 analytic-bound evaluations, not strict interval certificates.

Canonical entry commands (repository root):

```text
python -B -m unittest discover -s experiments -p 'test_round9*.py' -v
python -B experiments/round9_original_models.py --config experiments/configs/round9_local_cert_v1.json --checkpoint-root PATH_TO_ROUND6_MODELS --preflight
python -B experiments/round9_dispatch.py smoke --source-dir experiments --checkpoint-root PATH_TO_ROUND6_MODELS --run-dir PRIVATE_NEW_SMOKE
python -B experiments/round9_dispatch.py launch --source-dir experiments --checkpoint-root PATH_TO_ROUND6_MODELS --run-dir PRIVATE_NEW_RUN
python -B scripts/analyze_round9_endpoint.py --input-dir runs/round9_local_cert_v1_20261011T045942Z/snapshots/20261011T174341Z
```

The dispatcher records a durable CPU process receipt and enforces a one-hour
limit. It performs no training and never substitutes the separate width-16
Round 9 attachment fixture for an original checkpoint.

Round 7 completed on 2026-10-10 at 11:21:13 UTC in 2,776.51 seconds. It evaluates
float64 GS-Thin against mass-preserving uniform subsets, reusing all ten
qualified Round 6 histogram checkpoints without training. The question is how
prefix record budget relates to both directional sequence KL risks after
one-time compression and each decoder's own suffix-KV updates. The candidate
worst-case rate is `n*T/(m^2*(n+T))` in the restricted equal-weight subset class;
it is not a required exponent on trained models. Read the frozen
[Round 7 protocol](experiments/PROTOCOL_ROUND7_THINNING_V1.md),
[proof snapshot](theory/archive/round7/THEOREM_AND_PROOF.md), and
[run status](runs/round7_thinning_v1_20261010T102523Z/RUN_STATUS.md).
All 70,560 records passed server-side validation; all ten checkpoints passed
their frozen quality gate. GS-Thin mean KL is lower in 96/108 paired cells;
all 12 exceptions use n=16,m=2. No GS forward row meets the audited small-risk
condition (0/17,280). These are finite histogram results, not a general theorem
or useful uniform certificate. Round 6 and Experiment 0 evidence remain unchanged.

The completed compact endpoint is
`runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/`.
Start with [RESULTS.md](RESULTS.md). Reproduction commands are in
[README.md](README.md); offline publication verification is
`python -B scripts/verify_round7_release.py`. The public snapshot retains all
training-seed aggregate values and analytic CSVs, but excludes raw trained
rows, checkpoints and operational records. Its verifier does not reconstruct
row-level coverage or recalculate checkpoint norms.

The completed Round 6 question was whether one-time, positive mass-preserving prefix-KV
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
