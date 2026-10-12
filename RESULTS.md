# L01 results

## Round 11: trained-model temporal-policy mechanism

**Execution complete; both predeclared descriptive screens pass in 5/5 model
seeds.** The five original two-layer Round 6 histogram checkpoints were reused
without training. At identical suffix-pair record budgets, the temporal order
of thinning signs strongly changes estimated sequence risk. This is a finite
mechanism result; the earlier Round 9 half-cache certificate remains failed.

The run completed on 2026-10-12 at 04:28:23 UTC in 117.40 seconds including
analysis (115.46 seconds in the scientific runner). All 960 primary rows,
60 Fisher rows and 7,560 event rows are present. Start with the
[run interpretation](runs/round11_streaming_covariance_v1_20261012T042125Z/INTERPRETATION.md),
then the [generated per-seed report](runs/round11_streaming_covariance_v1_20261012T042125Z/RESULTS.md).

### Actual generated sequence risk

Each model uses four fixed n=64 prefixes and 128 token paths per policy.
Average schedules and prefixes within each training seed, then average the
five seeds. The estimand is the mean fixed-schedule sequence KL(Q||P), not
KL of the law after marginalizing the hidden compression schedule.

| Horizon | Coherent mean KL | IID mean KL | Alternating mean KL |
|---:|---:|---:|---:|
| 16 | 0.0235324 | 0.00968247 | 0.00495067 |
| 32 | 0.141446 | 0.0288152 | 0.0158924 |
| 64 | 0.694266 | 0.0795661 | 0.0431900 |
| 128 | 2.730185 | 0.175011 | 0.100431 |

At T=128 all five models have coherent > IID > alternating. The per-model
max/min ratio of policy mean direct local attention squared errors is
1.0055--1.0118, passing the predeclared <=1.25 screen. The paired
coherent-minus-alternating difference is 2.62975, with descriptive 95%
training-seed bootstrap interval [2.57981,2.68374]. Prompts, paths and
schedules are not additional independent models. The local screen compares
means, not full error laws or vectors. All three policies retain the old
prefix exactly and halve eligible newly generated suffix pairs; this is not
whole-cache half compression or a byte/throughput measurement.

### Fixed-reference cross-event Fisher diagnostic

On two prefixes per model, 16 P histories, T=32 and amplitude 0.25, the full
event-JVP Fisher quadratic predicts finite reverse KL(P||Q) much better than
its diagonal-only counterpart. These are averages of per-prefix relative
errors, followed by equal training-seed averaging.

| Policy | Full Fisher mean relative error | Diagonal-only mean relative error |
|---|---:|---:|
| Coherent | 0.04810% | 79.6573% |
| Alternating | 0.01206% | 78.4968% |

Both policies pass the predeclared error screen in every seed. The finite
KL and quadratic use the same P histories: this is an offline local
explanation, not heldout risk prediction. Primary Q||P uses T=128 and Q
histories; the diagnostic P||Q uses T=32 and P histories. Although their
local quadratic metrics agree at zero perturbation, this experiment does
not show that Fisher explains the complete long-horizon primary risk gap.

The original generated Fisher figure applied amplitude squared twice;
CSV metrics and gates were correct. Use the separately saved
[corrected Fisher figure](runs/round11_streaming_covariance_v1_20261012T042125Z/plot_correction/fisher_predictions.png)
and [correction record](runs/round11_streaming_covariance_v1_20261012T042125Z/plot_correction/README.md).
The frozen original figures and source copies remain available for audit.

The temporal-policy ordering transfers to these trained models. The
linear/logarithmic/bounded asymptotic rates of the analytical construction
were not tested: in particular alternating KL still increases across this
horizon grid. No general Transformer theorem, useful certificate, new
compressor, practical speedup or publication novelty is established.

The [frozen protocol](experiments/PROTOCOL_ROUND11_STREAMING_COVARIANCE_V1.md),
[summary](runs/round11_streaming_covariance_v1_20261012T042125Z/summary.json)
and [recorded-evidence audit](runs/round11_streaming_covariance_v1_20261012T042125Z/release_verification.json)
bind the result. The audit recomputes recorded CSV aggregates and gates,
coverage, identities and source hashes; it does not replay checkpoints or
verify the mathematics. Read the summary before the secondary event CSV.

```text
python -B scripts/verify_round11_release.py --input-dir runs/round11_streaming_covariance_v1_20261012T042125Z
```

## Historical Round 9: local bounds and directional follow-ups

**Execution: COMPLETE (330/330). Scientific verdict: the frozen half-cache
feasibility gate fails for every method; near-full clustering succeeds.**
The current local bound improves on the former global bound in some cases,
but it does not certify high fidelity at substantial compression in this study.

A completed one-case [slack/JVP diagnostic](runs/round9_directional_diagnostic_v1_20261011T183000Z/INTERPRETATION.md)
supports a first-order description. The subsequent
[analytic remainder attempt](runs/round9_directional_remainder_v1_20261012T024528Z/INTERPRETATION.md)
fails its small-KL feasibility target: its upper is 30.9056. These follow-ups
do not change the failed half-cache certificate verdict.

The original five qualified width-64, two-layer Round 6 histogram checkpoints
were reused without training. Their hashes, architecture and quality metadata
matched the frozen manifest. The CPU run finished on 2026-10-12 at 01:43:41
Asia/Shanghai in 67.78 seconds (66.94 seconds in the scientific runner).
The canonical [snapshot](runs/round9_local_cert_v1_20261011T045942Z/snapshots/20261011T174341Z/summary.json)
contains all seed summaries; [condition_metrics.csv](runs/round9_local_cert_v1_20261011T045942Z/snapshots/20261011T174341Z/condition_metrics.csv)
retains all 330 conditions. Read the [run status](runs/round9_local_cert_v1_20261011T045942Z/RUN_STATUS.md)
for provenance, checks and audit limits.

## Round 9 half-cache result

At n=64,m=32,T=4, each method has 15 conditions: five training seeds and
three fixed prefixes. The primary criterion requires upper KL(Q||P)<=0.1 on
at least two prefixes in at least four seeds. **Each method has 0/5 passing
seeds and 0/15 passing prefixes.** Training seed is the independent model unit;
15 prefixes are not 15 independent models.

| Method | Mean upper KL(Q||P) | Upper min–max | Mean exact KL(Q||P) | Upper <=0.1 | Upper <1 |
|---|---:|---:|---:|---:|---:|
| Positive agglomerative means | 8.47630 | 0.91898–36.62308 | 5.43662e-6 | 0/15 | 1/15 |
| GS-Thin | 37.46756 | 36.52528–38.13886 | 0.00338083 | 0/15 | 0/15 |
| Uniform mass-preserving subset | 37.46756 | 36.52528–38.13886 | 0.02216680 | 0/15 | 0/15 |

Agglomerative bounds are below the reference universal cap in all 15 cases;
five give a Pinsker TV upper estimate below one. This is a weak nontrivial
improvement, not the promised small-KL/high-fidelity certificate. None is
<=0.02 (the sufficient KL threshold for TV<=0.1). GS and uniform produce the
same loose bounds here despite different exact KL values. The mean reference
cap is 37.47213; merely being below this cap is not the primary gate.

## Round 9 controls and interpretation

- Full cache passes the identity control in all 30 cases.
- At n=64,m=63, agglomerative has mean upper KL 0.00659081, 15/15<=0.1,
  and 14/15<=0.02. This reduces prefix records by only 1.5625%.
- At n=64,m=48 (25% fewer prefix records), agglomerative has mean upper KL
  4.13743 and 0/15<=0.1, although 10/15 are below 1.
- At n=16,m=8, every method also has 0/15<=0.1.

The tiny observed clustering KL and much larger bound identify a limitation
of this sufficient certificate, not an observed failure of compression or a
refutation of a worst-case theorem. Half-cache zero-variance tube intersections
occur in 2/15 clustering cases and all 15 cases for each subset method. These
diagnostics are consistent with loose propagation control; they do not alone
identify the cause of all slack. Clustering uses synthetic positive weighted
means and lies outside the Round 7 equal-weight original-subset lower-bound class.

This is a finite T=4, three-token full-softmax evaluation: 40 prediction histories
and 81 leaves per condition. It establishes neither long-generation fidelity
nor general NLP behavior. The numbers are **float64 analytic-bound evaluations,
not outward-rounded interval certificates**; no throughput or memory-byte gain
was measured.

Independent record checks pass: all 330 expected keys are unique and present;
recorded model/source/config provenance, tree sizes, full-cache controls, both
sequence inequalities, seed/group arithmetic and primary gates agree. The
[verification record](runs/round9_local_cert_v1_20261011T045942Z/snapshots/20261011T174341Z/verification.json)
states the limits: omitted per-history files and weights were not independently
replayed, and this is not proof or novelty verification. Recheck the public
snapshot from the repository root:

```text
python -B scripts/analyze_round9_endpoint.py --input-dir runs/round9_local_cert_v1_20261011T045942Z/snapshots/20261011T174341Z
```

## Round 9 follow-up: frozen one-case slack and JVP diagnostic

Reuse the first original two-layer seed (11), first n=64 prefix, m=32 positive
agglomerative summary and T=4. All 40 histories are evaluated, with each theta
decoder generating its own suffix KV. This mechanical selection happens to be
the largest-upper-bound cluster case in the previous 15-row table. It is a
descriptive mechanism diagnostic, not an independent-model estimate.

| Diagnostic | Median over histories | Maximum |
|---|---:|---:|
| Hidden radius / actual hidden error | 166.42 | 295.07 |
| Final LN chord bound at actual error / actual LN error | 1.120 | 1.220 |
| Centered-head norm bound / actual centered logits | 5.748 | 15.952 |
| Exact-logit quadratic bound / actual conditional KL | 1.993 | 11.384 |
| JVP vector remainder / max(endpoint,JVP,1e-10) | 1.850% | 6.369% |

The <=25% worst-history first-order screen passes. Endpoint identity,
forward-JVP/finite-difference checks and replay of the selected Round 9 row pass.
Actual hidden errors are 0.0380–0.1233 versus radii 4.430–36.091. The result
locates substantial upstream propagation slack; output direction also matters,
but the head's rank alone does not explain the entire gap. The stage ratios
are not an exact product because the old and actual radii induce different
LayerNorm gains.

The existing P-only sequence bound is 36.623076, exact KL is 3.45822e-6.
Q-weighting the existing conditional bounds still gives 21.605575. Replacing
their hidden radii by actual Q hidden-error norms gives 1.56286e-4 with the
existing scalar readout machinery. This is an oracle substitution using Q
states and Q history masses, **not a new reference-only certificate**.

Read the [interpretation](runs/round9_directional_diagnostic_v1_20261011T183000Z/INTERPRETATION.md),
[generated measurements](runs/round9_directional_diagnostic_v1_20261011T183000Z/RESULTS.md),
then [summary](runs/round9_directional_diagnostic_v1_20261011T183000Z/summary.json).
The raw per-history vectors are secondary evidence. Four targeted tests pass;
CPU calculation took 3.48 seconds. The [offline verification](runs/round9_directional_diagnostic_v1_20261011T183000Z/verification.json)
checks source copies, vectors, ratios, aggregates and tree masses; it does not
replay the checkpoint or certify a second-derivative supremum. One successful
JVP diagnostic does not identify a unique cause, establish generalization, or
change the Round 9 feasibility gate.

## Round 9 next step: analytic directional remainder attempt

The same seed-11, first-prefix n=64,m=32,T=4 case was evaluated with signed
P-path tangents and analytic second-derivative norm enclosures on theta in
[0,1]. Construction uses P and the fixed summary, not actual Q endpoint states;
each path generates its own suffix KV.

| Quantity | Result |
|---|---:|
| New sequence upper, intersected with original conditional bounds | 30.905579 |
| Original sequence upper | 36.623076 |
| Target<=0.1 | FAIL |
| Raw directional sequence upper | 3.43908e17 |
| Improved conditional bounds | 27/40 |
| Finite second-derivative enclosures | 40/40 |
| Median / maximum Taylor-remainder norm upper | 1.68686 / 1.17287e9 |

The prior accurate JVP is reproduced to <=4.52e-16. Its small measured remainder
does not imply a tight computable supremum bound. On history (0,2), the enclosure
grows through second-block attention and MLP, then final LayerNorm reaches
2.55604e9. Rare continuation-symbol histories give extreme bounds, but the
empty-history conditional upper is already 3.32726. This is not solely a
rare-event or history-law-transfer issue.

Read the [interpretation](runs/round9_directional_remainder_v1_20261012T024528Z/INTERPRETATION.md),
[generated report](runs/round9_directional_remainder_v1_20261012T024528Z/RESULTS.md),
then [summary](runs/round9_directional_remainder_v1_20261012T024528Z/summary.json).
The [method note](theory/ROUND9_DIRECTIONAL_REMAINDER_V1.md) derives the inequalities.
Four focused tests and the [offline record audit](runs/round9_directional_remainder_v1_20261012T024528Z/verification.json)
pass; CPU scientific calculation took 1.36 seconds. This is a bounded negative
result for norm-only second-order enclosures, not impossibility of every
directional method or a contradiction of earlier theory. It adds neither
a half-cache certificate nor publication novelty. All numerical bounds remain
float64 evaluations, not outward-rounded certificates.

## Previous Round 7 completed thinning study

**Execution: COMPLETE. Scientific interpretation: GS-Thin usually improves
finite histogram fidelity; the horizon coordinate is informative, but the
global small-risk certificate remains vacuous.** Neither publication-level
novelty nor a general trained-Transformer scaling law is established.

The canonical Round 7 run is `round7_thinning_v1_20261010T102523Z`; its completed
endpoint is the [20261010T115125Z snapshot](runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/publication_manifest.json).
It finished on 2026-10-10 at 11:21:13 UTC in **2,776.51 seconds (46 min 17 s)**.
No models were trained: all ten qualified Round 6 histogram checkpoints were reused.

## Canonical Round 7 endpoints

| Component | Completed evidence | Interpretation |
|---|---|---|
| Trained evaluation | 70,560/70,560 valid records; 252 aggregate groups; no missing, duplicate or unexpected keys in server validation | Complete frozen grid, with separate KL directions |
| Checkpoint quality | 10/10 pass the frozen Bayes NLL gap gate; depths 2/4, five seeds each | Qualified for the synthetic histogram task |
| Paired GS-Thin versus uniform subset | GS lower mean KL in 96/108 cells; higher in 12/108 | All 12 exceptions are n=16,m=2; n=64/128 favor GS in all 72 cells |
| Paired uncertainty | 96 intervals below zero, six crossing zero, six above zero for GS minus uniform | Five training-seed bootstrap, 108 comparisons unadjusted for multiplicity |
| Exact analytic study | All recorded gates pass; exhaustive feasible summaries through T=128; one exact witness through T=2048 | Restricted equal-weight construction, not an arbitrary-compressor lower bound |
| GS forward small-risk certificate | **0/17,280** Q-to-P rows meet V<1/4; log V is **161.01–370.34** | No useful global certificate; this is not an observed-KL violation |

The trained grid uses n=16/64/128, T=8/32/128 and m/n=1/8,1/4,1/2, with eight
held-out prefixes, eight compression seeds and 16 sampled paths per direction.
The primary independent unit is the **training seed**, seeds 11/23/37/51/71.
Within each seed, paths are averaged, compression draws are averaged inside
each prompt, then prompts receive equal weight. Percentile 95% bootstrap
intervals resample the five training seeds. Grid cells share models and are
correlated; 96 favorable cells are not 96 independent discoveries.

Each summary defines its own law Q_R. The estimands are E_R KL(Q_R||P) and
E_R KL(P||Q_R), with paths sampled from the law in the first KL argument.
The shadow decoder receives the same sampled tokens but appends its own KV.
These are per-summary risks, not the KL of a compression-seed mixture.

## What Round 7 adds

At T=128 and m/n=1/4, representative Q-to-P sequence KL means are:

| Depth | n | m | GS-Thin | Uniform subset | KL reduction |
|---|---:|---:|---:|---:|---:|
| 2 | 64 | 16 | 0.14224 | 0.50947 | 72.1% |
| 2 | 128 | 32 | 0.06304 | 0.30635 | 79.4% |
| 4 | 16 | 4 | 0.19387 | 0.37702 | 48.6% |
| 4 | 64 | 16 | 0.07074 | 0.20999 | 66.3% |
| 4 | 128 | 32 | 0.02779 | 0.15085 | 81.6% |

The failure case is material. At depth 4, n=16,m=2,T=128, Q-to-P KL is
**0.832 versus 0.603**, with paired GS-minus-uniform difference **+0.229**
(95% interval **[+0.187,+0.269]**). At depth 4 all six n=16,m=2 intervals
favor uniform; at depth 2 all six corresponding intervals cross zero.
The gain is not universal, especially at the smallest retained budget.

The horizon coordinate h=nT/(n+T) explains the finite curves well. Across the
36 fixed depth/n/m/direction GS curves, the median max-to-min variation of
KL/h over T=8,32,128 is **1.059** (largest **1.205**). At budget fraction 1/4,
increasing T from 32 to 128 multiplies KL by **1.33–1.47** for n=16,
**2.03–2.12** for n=64, and **2.53–2.61** for n=128; h predicts 1.33, 2.0,
and 2.5 respectively. This supports slowing accumulation over the tested
window, not an asymptotic plateau or a general trained-model theorem.

Both methods' mean KL decreases with budget throughout the grid. Three
budgets do not establish an m^-2 trained-model law. The m^-1 and m^-2 figure
guides are visual references. Uniform sampling also has the finite-population
factor (n-m)/(m(n-1)); its slope need not equal -1 over these budget fractions.

All analytic gates pass. Exact minima over feasible equal-weight summaries
through T=128, the exact all-horizon minimum-gap witness, and balanced-uniform
expected risks are separate endpoints. At T=512/2048, the aggregate hard-class
and balanced-uniform values are curvature envelopes, not exact minimax or
measured KL. Analytic success checks the named construction and arithmetic;
it is not independent proof or novelty verification.

The main unresolved issue remains the global sufficient constants. Observed
GS Q-to-P row KL spans 0.0000114–10.239, but no row meets the audited small-risk
condition. Empirical MMD and sampled moment errors cannot replace the theorem's
uniform assumptions. Layer-level MMD and suffix drift are descriptive diagnostics,
not calibrated predictors or certificates of sequence KL.

## Round 7 evidence and reproduction

1. [Completed generated report](runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/analysis/RESULTS.md),
   [budget figure](runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/analysis/budget_scaling.png),
   [horizon figure](runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/analysis/horizon_scaling.png),
   and [multilayer figure](runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/analysis/multilayer_fidelity.png).
2. [Seed-level aggregates](runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/analysis/aggregates.json),
   [analytic summary](runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/analytic/summary.json),
   and [checkpoint hashes / norm audits](runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/checkpoints.json).
   Read the Markdown first; the 1.1 MB aggregates are secondary evidence.
3. [Protocol](experiments/PROTOCOL_ROUND7_THINNING_V1.md),
   [source map](GPT_CONTEXT.md), [theory qualifications](THEORY_STATUS.md), and
   [reproduction / verification](README.md). The publication manifest binds
   19 frozen source/configuration files and 17 compact endpoint files.

Raw trained JSONL, model weights and private operational records are excluded.
Offline verification reconstructs seed-level means and paired bootstrap
intervals, not raw-row coverage or norm audits from checkpoint bytes.
The only compressed records are the old prefix; the exact suffix grows with T.
No systems speedup, end-to-end memory advantage, arbitrary positive-weight
lower bound, uniform-generation certificate or publication readiness follows.
Round 6 and Experiment 0 numerical evidence below is unchanged.

## Previous Round 6 completed toy study

**Execution: COMPLETE. Scientific interpretation: analytic consistency and
qualified histogram evidence; recall training is unreliable and the global
certificate is uninformative.** This is not a general trained-Transformer
validation or evidence of publication readiness.

The formal run is `round6_toy_v1_20261010T061030Z`. The completed endpoint is
the [20261010T073818Z snapshot](runs/round6_toy_v1_20261010T061030Z/snapshots/20261010T073818Z/scientific_summary.json),
not the earlier launch-time analysis in the run's top-level `analysis/`.
Experiment 2 completed in 3,779.20 seconds (about 63 minutes); final analysis
finished on 2026-10-10 at 07:16:53 UTC.

## Canonical Round 6 endpoints

| Component | Completed evidence | Interpretation |
|---|---|---|
| Exact analytic construction | 160 configurations; 40 short-horizon sequence enumerations; all 12 recorded gates pass | Supports the implemented epsilon-squared and nT/(n+T) scales |
| Trained evaluation | 20/20 models; 22,080/22,080 prompt rows; 276/276 complete groups | All five training seeds per task/depth retained; no missing groups |
| Histogram training | 10/10 pass; Bayes NLL gap 0.000181–0.000473 versus threshold 0.03 | Competent models for this simple synthetic task |
| Associative recall training | 3/10 pass: 1/5 at depth 2, 2/5 at depth 4; threshold answer top-1 0.90 | Seven failures prevent a robust learned-recall interpretation |
| Global sufficient bound | No tighter than the model-specific universal bounded-logit cap in 840/840 applicable group-seed cases | No useful uniform-generation certificate; Pinsker reaches 1 in all 840 cases |

The primary statistical unit is the **training seed**, with five seeds per
task/depth. KL is the sum of conditional Q-to-P KL on Q-sampled histories,
averaged over eight paths per prefix, 16 prefixes per seed, then seeds. It
estimates sequence KL for the fixed empirical prompt set. Error bars are
two-sided 95% Student-t intervals across the five seeds. The rows and paths
are not independent training replicates. Failed-quality seeds are included
in every primary aggregate; no outcome-dependent retuning was performed.

## What the curves show

For the exact construction, `KL / (epsilon^2 * n*T/(n+T))` ranges from
**0.110134 to 0.124873** across all 160 configurations. The finite sums display
short-horizon growth and long-horizon saturation. This validates the named
construction and its arithmetic, not typical trained-model behavior or a
minimax cache-space lower bound.

The trained histogram results show slowing accumulation over the measured
horizons. Representative five-seed mean sequence KL below uses prefix n=64,
summary m=16, rho=1; both summaries preserve mass.

| Histogram depth / method | T=8 | T=32 | T=128 |
|---|---:|---:|---:|
| 2 layers / uniform weighted subset | 0.05956 | 0.18347 | 0.39229 |
| 4 layers / uniform weighted subset | 0.06108 | 0.19372 | 0.42458 |
| 2 layers / positive cluster | 0.00001274 | 0.00003532 | 0.00006847 |
| 4 layers / positive cluster | 0.00000500 | 0.00001494 | 0.00003239 |

For n=64, the coordinate nT/(n+T) is 7.11, 21.33 and 42.67 at these horizons.
These finite curves are consistent with slowing accumulation; T<=128 does
not establish the eventual plateau. The very small cluster-summary errors
are descriptive evidence for this task and compressor, not a general bound.

The rho-mixture precision diagnostic is also informative on histogram. Across
depths and horizons, mean KL at rho=0.25 is **0.0595–0.0618** times its rho=1
value, and at rho=0.5 is **0.2412–0.2480** times it, close to rho squared.
These mixtures store **n+m=80** prefix records per layer rather than m=16;
rho is intervention amplitude, not a certified all-query epsilon. Recall
ratios are less consistent and most recall models failed the training gate.

The global audit records log C_model from **139.22 to 1869.95** and log coarse
uniform epsilon from **12.70 to 261.61**. The theorem may be valid while these
global sufficient constants have no useful explanatory force in this setting.
Observed-query numerator/denominator errors remain sampled diagnostics;
substituting them into the theorem would not certify its assumptions. No
constant was fitted to observed KL.

## Evidence and reproduction route

1. Read the [completed derived report](runs/round6_toy_v1_20261010T061030Z/snapshots/20261010T073818Z/analysis/RESULTS.md)
   and [three figures](runs/round6_toy_v1_20261010T061030Z/snapshots/20261010T073818Z/analysis/multilayer_autoregressive_fidelity.png).
   Figure 3 retains both 25% and 50% budget curves for each method and labels
   failed task-quality coverage. The figure's t intervals can extend below zero;
   KL estimates themselves are nonnegative up to numerical tolerance.
2. Use [scientific_summary.json](runs/round6_toy_v1_20261010T061030Z/snapshots/20261010T073818Z/scientific_summary.json)
   for all model quality records and planned/observed coverage;
   [analysis/aggregates.json](runs/round6_toy_v1_20261010T061030Z/snapshots/20261010T073818Z/analysis/aggregates.json)
   contains seed-level curves, uncertainty and bound status. Do not open the
   large aggregates or compressed per-prompt endpoints first.
3. [Snapshot provenance](runs/round6_toy_v1_20261010T061030Z/snapshots/20261010T073818Z/provenance.json)
   binds frozen source/configuration and allowlisted endpoint files. The
   [frozen protocol](experiments/PROTOCOL_ROUND6_TOY_V1.md) defines tasks,
   compression, gates and sampling. [README.md](README.md) gives reproduction.

Offline verification `python -B scripts/verify_round6_release.py` passes:
12 frozen source hashes, three endpoint-file hashes, eight derived-analysis
hashes, complete coverage and group aggregation, 2,880 full-cache identity
rows, and the recomputed 160-row analytic grid. The verifier checks recorded
norm-audit fields but does not recalculate norms from private checkpoints.

Actual Q cache tensor bytes include suffix K/V and explicit weights, but exclude
allocator residency and Python overhead. Dense replay/oracle storage is not Q's
compression cost. No practical speedup, end-to-end memory saving, fixed-memory
streaming or full-sequence TV endpoint is established. The suffix grows with T.
The unweighted subset is an out-of-theorem mass-changing control.

The next empirical gap is a reliably learned recall baseline under a new frozen
protocol, preserving these failures. A useful model-dependent propagation
certificate is a separate theoretical gap. Neither is resolved by the current
run. Historical Experiment 0 remains unchanged below.

---

# Experiment 0 result

**Execution: COMPLETE, 144/144 cases. Scientific endpoint:
NO_HELDOUT_SUPPORT.** Adding recent-position information did not improve the
prespecified heldout future-KL prediction objective.

## Primary endpoint

All models below are fixed ridge regressions calibrated on 8 articles. Evaluation
uses 16 different articles, each with 6 interventions. Loss is squared error in
log10 future cumulative conditional KL, averaged within each article and then
across articles. Lower is better.

| Predictor | Heldout prediction MSE | Case-level Spearman |
|---|---:|---:|
| Current direct-residual KL proxy | 0.1040362242 | 0.23936 |
| Current plus recent mean/max proxies | 0.1042276286 | 0.24259 |
| Exact current KL, separately calibrated reference | 0.0949199813 | 0.20129 |

The primary baseline-minus-candidate **loss reduction** is
**−0.0001914044**, or **−0.18398%** improvement. Its paired article-bootstrap
95% interval is **[−0.0004202861, +0.0000445716]** in loss units. The frozen gate
required at least 10% reduction and a positive lower bound; it was not met.
The small observed worsening is not evidence of a statistically established
worsening. The bootstrap resamples articles while conditioning on the one
sampled continuation in each intervention cell; it does not quantify additional
rollout-sampling variance. Spearman is descriptive; no cell-level significance
is claimed.

The `actual_current_oracle` key in the raw summary means **exact current KL**.
It has no future information and is not a future oracle. It is an additional
reference, not the primary gate or a deployable cheap method.

## What ran

| Item | Recorded setting |
|---|---|
| Frozen model | Qwen2.5-7B-Instruct, BF16, eager attention |
| Inputs | 24 distinct WikiText validation articles; 8 calibration / 16 heldout |
| Context | At most 384 tokens including the chat template |
| Intervention | Persistent logical deletion of 20% of eligible entries at one layer |
| Layers | 5, 14, 22, zero-based |
| Masks | Seeded random; lowest current mean-head attention |
| Future horizon | 32 prediction positions, excluding current t=0 |
| Sampling | Temperature 1, full softmax, no top-k/top-p or EOS stopping |
| Completed cells | 48 calibration + 96 heldout = 144 |
| Runtime | 157.13 seconds including worker startup and final analysis |
| Hardware | One NVIDIA GeForce RTX 5090 |

Q samples the continuation with its masked cache. P replays exactly those tokens
with its own unmodified cache; KL direction is Q || P at each shared prefix.
The target sums positions t=1 through t=32: the final distribution is evaluated
after the 32nd sampled token. It is not the KL of exactly the sampled 32-token
string, and it is not an empirical sequence-TV measurement.

## Interpretation and limits

This run gives no heldout support to the **four-token chat-suffix,
direct-residual proxy variant**. The runner selects the final four positions
after applying the chat template. The pinned model's
[template](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/a09a35458c702b33eeacc393d103063234e8bc28/tokenizer_config.json)
ends the user turn and appends an assistant header. These are boundary/header
query positions, not the last four article-content queries. Sparse large proxy
values also dominate the raw recent-window mean; regression uses log-transformed
features. The proxy omits downstream attention and MLP responses. These limit
which mechanism has been tested.

The result does not disprove future observability, establish that current KL is
universally sufficient, or measure the spectrum of a causal observability
Gramian. There is one model, one prompt family, one eviction fraction and one
sampled continuation per intervention. No stable low-sensitivity subspace or
uniform stability certificate has been established.

Masking retains tensor storage. GPU allocation includes model weights and the
diagnostic harness; it is not evidence of cache compression. The scoring method
is inspired by a direct-residual approximation, not a replication or comparison
against the complete Behavior-Preserving compression algorithm. No quality,
memory or speed superiority over that method is claimed.

The decision is to stop this proxy candidate under the frozen protocol. A
different proxy requires a new rationale, protocol and run, preserving this
negative result.

## Evidence

Canonical run: `runs/observable_gain_v1_20261009T190129Z`.

- [Analysis aggregate](runs/observable_gain_v1_20261009T190129Z/observable_gain_summary.json)
- [Execution and hash summary](runs/observable_gain_v1_20261009T190129Z/execution_summary.json)
- [Runtime aggregates](runs/observable_gain_v1_20261009T190129Z/summary.json)
- [Frozen generated report](runs/observable_gain_v1_20261009T190129Z/RESULTS.md)
- [Secondary per-case evidence](runs/observable_gain_v1_20261009T190129Z/cases.jsonl)

Offline replay, trajectory checks and input/source hashes are checked by
`scripts/verify_results.py`. Two focused analysis tests check calibration-only
fitting, split isolation and recovery of a deterministic feature advantage.
The model revision comes from the frozen configuration and launch record;
independent byte hashes of the model weights and tokenizer were not recorded.
The separate GPU smoke checked cache replay identity and attention deletion
semantics; it contributed no scientific data.
