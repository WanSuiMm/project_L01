# Incremental review: Round 9 directional follow-ups

Review base: `e6281c76f25a5c14968a4752e254b8ca24eea058`.
Evidence head: `ce763360de90a535d13797b9206d4e0634c70f8c`.
This later commit changes only this handoff; the evidence head stays fixed.

## Minimal reading order

1. [RESULTS.md](RESULTS.md), sections "Round 9 follow-up" and "Round 9 next
   step", for the new measurements and negative verdict. Earlier aggregate
   evidence is unchanged and need not be reread first.
2. [Directional diagnostic interpretation](runs/round9_directional_diagnostic_v1_20261011T183000Z/INTERPRETATION.md),
   followed by [remainder interpretation](runs/round9_directional_remainder_v1_20261012T024528Z/INTERPRETATION.md).
3. [Analytic method note](theory/ROUND9_DIRECTIONAL_REMAINDER_V1.md), then
   [GPT_CONTEXT.md](GPT_CONTEXT.md) for exact source symbols and limits.
4. Each new run's `summary.json`, `checks.json`, `verification.json`,
   `provenance.json`, `checkpoint.json` and `source/` bind the records.
   Full `histories.jsonl` vectors/stages are secondary evidence; do not open
   them before the interpretations and summaries.

## Changed evidence and claims

Both follow-ups freeze the original seed-11 two-layer, width-64 histogram
model and first n=64 prefix (`histogram-n64-0000`), positive agglomerative m=32
summary, T=4, all 40 prediction histories and 81 leaves. Each interpolation
path generates its own suffix KV at the original absolute positions. There
is no retraining, new model grid, prefix search or budget/horizon sweep.
This mechanical choice happens to have the largest original clustering
upper among 15 cases; it is not a representative-model estimate.

The diagnostic finds hidden-radius / actual-error median 166.42x versus
additional centered-head slack median 5.75x. Full-unroll JVP scaled vector
remainder is median 1.85%, maximum 6.37%, passing the frozen 25% screen.
Actual Q states and Q history masses enter these oracle diagnostics; they
cannot be substituted into a reference-only certificate.

The next attempt retains signed first-order directions and encloses
second-derivative norms on theta in [0,1]. Construction uses P and the fixed
summary; actual Q endpoints are used only for separate recorded validation.
Full-sequence aggregation uses the P tree with the stated entropy-transfer
bound or maximum path risk, not a direct P-weighted risk average.

| Quantity | Result |
|---|---:|
| Original sequence KL upper | 36.623076 |
| New upper intersected with original conditional bounds | 30.905579 |
| Frozen target <=0.1 | FAIL |
| Actual sequence KL(Q||P), validation only | 3.45822e-6 |
| Raw directional sequence upper | 3.43908e17 |
| Improved conditional bounds / finite enclosures | 27/40 / 40/40 |
| Empty-history conditional upper | 3.327256 |

Large second-order enclosures grow through the second block and final
LayerNorm. The root already exceeds the target, so rare histories and history
law transfer are not the sole failure. Exact internal LayerNorm denominator
contributions were not retained. Accurate endpoint linearization does not
establish a tight second-derivative supremum enclosure.

## Unchanged claims and research decision

The original 330-condition Round 9 endpoint still has 0/5 passing models and
0/15 passing prefixes at half cache for each method. Near-full clustering
m=63 passes but retains 63/64 records. Round 6, Round 7 and Experiment 0
evidence and claims are unchanged.

No useful half-cache certificate, strict interval certificate, general
Transformer guarantee, systems saving or publication novelty is established.
This one-case failure does not prove all directional methods impossible or
refute the earlier conditional theory. The current recommendation is to stop
expanding this scalar second-order enclosure into larger sweeps; another
attempt would require a concretely different remainder mechanism.

## Verification actually recorded

Eight focused software tests (four per follow-up) pass. Both offline audits
pass for frozen hashes, all 40 histories, vector/Taylor arithmetic, aggregate
counts and verdicts. The remainder audit reconstructs all 81 path risks and
the fixed-lambda entropy-transfer comparison. It does not replay weights,
prove derivative suprema or perform outward-rounded arithmetic. All reported
bounds are ordinary float64 analytic evaluations.

Run from the repository root without checkpoint weights:

```text
python -B scripts/verify_round9_directional_diagnostic.py --input-dir runs/round9_directional_diagnostic_v1_20261011T183000Z
python -B scripts/verify_round9_directional_remainder.py --input-dir runs/round9_directional_remainder_v1_20261012T024528Z
```

CPU software tests require the dependencies in `requirements.txt`:

```text
python -B -m unittest discover -s experiments -p 'test_round9_directional_*.py' -v
```

Exact scientific replay requires the separately held original checkpoint;
weights and operational receipts are excluded. Commands in `PROJECT.md`
require a fresh output directory and do not overwrite this evidence.

## Concrete reviewer questions

1. Do the attention denominator and quotient remainder inequalities cover
   zero endpoint prefix weights, mixed query/key/value derivatives and each
   path's self-generated suffix KV without using actual Q measurements?
2. Is the sequence aggregation valid for KL(Q||P), including its old-bound
   conditional intersections, without replacing Q history weights by P?
3. Does the interpretation distinguish accurate endpoint JVP, failed
   computable remainder enclosure and untested alternatives, with the
   conclusion confined to this frozen case and method?
