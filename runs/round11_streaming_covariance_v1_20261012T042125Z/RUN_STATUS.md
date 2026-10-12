# Round 11 completed run

- Protocol: `round11_streaming_covariance_v1`.
- Completion: 2026-10-12 04:28:23 UTC; exit zero, automatic analysis complete.
- Scientific runner: 115.46 seconds; complete dispatch including analysis: 117.40 seconds.
- Execution: CPU, float64, two threads; no training.
- Coverage: 960/960 primary rows, 60/60 Fisher rows, 7,560/7,560 event rows.
- Original checkpoint hashes and quality metadata matched for all five models.
- Ordering and mean-local-distortion screens: 5/5; local Fisher screen: 5/5.

The four targeted streaming-core software tests passed before launch.
Original seed-11 smoke and analyzer checks passed locally and on the
execution server. These check records were retained operationally rather
than embedded in the frozen scientific provenance, whose software-check
list is empty. Model identity, antipodal derivative and physical-endpoint
parity gaps are recorded in the Fisher CSV (maxima below 5e-15).

The completed scientific directory was retrieved intact; generated CSV,
JSON, source copies and figures were not regenerated or edited. Added
interpretation, status and release-verification records are separate from
those frozen outputs. Operational PIDs, private paths and launch logs are
excluded from the public release; progress.json is also excluded.

Read [INTERPRETATION.md](INTERPRETATION.md), then [RESULTS.md](RESULTS.md).
Run the read-only audit from the repository root:

```text
python -B scripts/verify_round11_release.py --input-dir runs/round11_streaming_covariance_v1_20261012T042125Z
python -B -m unittest discover -s experiments -p test_round11_streaming_core.py -v
```

This audit recomputes recorded coverage, seed aggregates, relative-error
arithmetic, gates and source hashes. It does not replay checkpoint weights,
omitted per-token paths, derivatives, proofs or asymptotic laws. Exact
scientific replay requires the separately held original checkpoints;
their hashes and metadata are public, their weights are not distributed.
