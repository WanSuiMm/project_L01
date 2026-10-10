# Round 6 formal run

## Completed endpoint (2026-10-10)

**Execution COMPLETE:** 20/20 models, 22,080/22,080 prompt rows, 276/276
groups, no partial or missing groups. Transformer execution took 3,779.20
seconds; final analysis finished at 07:16:53 UTC.

Read the [canonical completed report](snapshots/20261010T073818Z/analysis/RESULTS.md)
and [project interpretation](../../RESULTS.md) first. The
[scientific summary](snapshots/20261010T073818Z/scientific_summary.json)
retains all training seeds, including seven recall quality failures.
Histogram qualified 10/10; recall qualified 3/10; the coarse global bound is
uninformative relative to the universal bounded-logit cap. Runtime completion
does not imply a scientific pass.

The record below describes dispatch. Its top-level `analysis/` is a preserved
earlier snapshot, not the completed trained endpoint.

## Original dispatch record

Run: `round6_toy_v1_20261010T061030Z`. Formal dispatch: 2026-10-10 06:13:51 UTC.

**Experiment 1 complete; Experiment 2 dispatched and CUDA training verified.**
This is a launch-time record, not ongoing monitoring or a completed trained
Transformer result.

Experiment 1 completed 160 exact analytic configurations, with 40 short-horizon
binary sequence enumerations. All 12 recorded gates passed. The ratio
`KL / (epsilon^2 * n*T/(n+T))` spans 0.1101336577 to 0.1248728361.
These results validate the named analytic construction and implementation,
not typical trained-model behavior or minimax cache space.

Start with [analysis/RESULTS.md](analysis/RESULTS.md) and
[experiment1/summary.json](experiment1/summary.json). The first two figure pairs
are analytic results. The local multilayer figure is a pending-data panel;
no trained-model claim is made. Large per-prompt traces should be read after
the completed trained-model aggregates become available.

Experiment 2 plans 20 independently trained checkpoints: two tasks, two
depths and five seeds. It uses prefix lengths 16/64/128, nested generation
horizons 8/32/128, 16 held-out prefixes and 8 Q continuations per prefix.
Full, uniform weighted subset, positive cluster and unweighted subset controls
are frozen. All trained seeds remain in the result, including quality failures.

The server supervisor automatically produces all three figure pairs after
Experiment 2 completes. Admission can wait up to 24 hours; execution is capped
at 12 hours, with partial evidence retained as incomplete. At dispatch a GPU
met the admission rule and training began.

Read the [frozen protocol](../../experiments/PROTOCOL_ROUND6_TOY_V1.md).
Source and configuration bindings are in [provenance.json](provenance.json).
Operational host/PID/path receipts are stored privately.
