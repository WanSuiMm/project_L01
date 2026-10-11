# Round 9: local bounds on the original Round 6 two-layer models

Status: frozen prospective protocol; no original-model endpoint has been run.

Question: can reference-only local propagation yield an informative finite-law
KL upper bound when substantially reducing the old-prefix cache of the already
qualified Round 6 histogram decoders? No training or checkpoint substitution is
allowed. This is an evaluation of analytic bounds in float64, not an outward-
rounded, machine-certified enclosure.

## Frozen inputs and interventions

Use all five original width-64, two-layer, feed-forward-multiplier-4 histogram
models (training seeds 11, 23, 37, 51, 71). The expected byte hashes and recorded
quality results are copied from the completed Round 7 checkpoint manifest.
Reject absent, mismatched or unqualified checkpoints; do not retrain replacements.
Check the explicitly traced decoder against the original Round 6 implementation.

Generate three fixed prefixes for each n in {16, 64} with the existing Round 6
`make_prompts` / Beta(2,2)-Bernoulli sampler. Its seed is
`stable_seed("round9-prefix", prefix_seed, n)`. All models and methods use the
same six prefixes. Record the tokens and their hashes before evaluation.

The horizon is T=4 with temperature 1 and the complete three-token softmax,
including CONT. There are 40 prediction histories and 81 complete sequences.
Compress only the old prefix once; each model then appends its own suffix KV
on each common hypothetical token history. No future outcome selects a summary.

At each n use a full-cache control, a near-full budget m=n-1, and retained
fractions 0.75, 0.5, 0.25 (25%, 50%, 75% old-prefix record reductions).
Agglomerative positive cardinality-weighted means and uniform mass-preserving
original-record subsets run at all four compressed budgets. GS-Thin runs at
0.5 and 0.25 only because repeated halving requires n/m to be a power of two.
The uniform and GS RNG for a layer is
`stable_seed("round9-summary", summary_seed, prompt_id, n, layer)`.
Every summary has exactly m positive-weight records and total prefix mass n.
Synthetic cluster means are outside the Round 7 original-subset lower-bound class.

There are 11 conditions per model/prefix: one full, eight cluster/uniform and
two GS conditions. Total: 5 models x 6 prefixes x 11 = **330 condition rows**.
One summary seed is frozen; there is no best-of-random-seeds selection.

## Bound and independent measurement

`round9_certify.py` contains the attachment's finite local inequalities. Its
operational copy changes checkpoint loading to weights-only and includes final
LayerNorm in tube statistics. The original attachment remains private/read-only.

`round9_original_models.py` shares the reference P tree across summaries.
The certificate uses only P trajectories, the fixed prefix summary and recursively
bounded state perturbations. It computes both the worst-path bound and the
fixed-lambda entropy-transfer bounds on the attachment's frozen lambda grid;
their minimum is a valid analytic upper bound, not the continuous optimum.
Separately, a Q tree computes both directional sequence KL values using the
chain rule. Q does not enter the construction of the reference-only bound.
Local violations, nonfinite quantities, lost mass or incomplete trees fail the
run rather than being silently dropped. A no-Q replay is checked in the smoke.

The shared-tree implementation and the attachment replay must agree on a
bounded fixture before deployment. This is a software check, not a scientific
endpoint. Original-model implementation agreement is checked in the remote smoke.

## Outcomes and predeclared decision

Primary metric: the full-tree upper bound on D_KL(Q||P), by model, prefix,
method and budget. Secondary metrics: exact finite-tree Q||P and P||Q, reverse
bound, bound/actual ratio, Pinsker TV upper estimate capped at one, reference-only
trivial cap, forcing and propagation radii, LayerNorm zero-variance intersections
including final LayerNorm, and time. All numeric bounds retain their float64 label.

Report threshold fractions at upper <=0.02 (sufficient TV<=0.1), <=0.1, and <1.
Being below a trivial cap is recorded separately; it is not a high-fidelity gate.
The primary budget is n=64,m=32. A method passes the prospective **numerical
feasibility** gate if at least four of five training seeds have upper <=0.1 on
at least two of their three fixed prefixes. Evaluate each method separately.
This finite gate does not establish population coverage or strict certification.
Near-full success alone does not qualify substantial compression.

Training seed is the independent model unit. Report all five seed summaries,
including failures; prefixes are secondary variation. Do not treat 40 histories,
81 leaves or repeated timings as independent model samples. No bootstrap claim
is required for this bounded qualification.

If no method passes at 50% reduction, retain the negative result and diagnose
where the analytic slack enters before proposing directional verification.
Scalar-ball failure alone does not identify its cause or establish novelty.

## Execution and provenance

Use a fresh directory and the registered server Python, CPU float64, two BLAS/
Torch threads, and a one-hour hard limit. No GPU is required. Launch through
`round9_dispatch.py` so the job survives disconnect and records a private
host/PID/command/source-hash receipt. Existing checkpoints and runs are read-only.
Do not overwrite outputs or automatically restart partial runs.

From the repository root:

```text
python -B -m unittest discover -s experiments -p 'test_round9*.py' -v
python -B experiments/round9_original_models.py --config experiments/configs/round9_local_cert_v1.json --checkpoint-root PATH_TO_ROUND6_MODELS --preflight
python -B experiments/round9_dispatch.py smoke --source-dir experiments --config configs/round9_local_cert_v1.json --checkpoint-root PATH_TO_ROUND6_MODELS --run-dir PRIVATE_NEW_SMOKE
python -B experiments/round9_dispatch.py launch --source-dir experiments --config configs/round9_local_cert_v1.json --checkpoint-root PATH_TO_ROUND6_MODELS --run-dir PRIVATE_NEW_RUN
```

Each case stores a compact row and per-history bounds/probabilities in a secondary
file. `summary.json` and `RESULTS.md` are the reading entry points; large per-
history files are for reconstruction. Bind source/config/expected-checkpoint
hashes and actual checkpoint hashes. Private operational records and weights
remain excluded from public delivery. No memory-byte or inference-speed benefit
is measured by this protocol.
