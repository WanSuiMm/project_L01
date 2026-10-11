# Round 9 original-model certification: completed endpoint

Status: **COMPLETE / HALF_CACHE_FEASIBILITY_FAIL / NEAR_FULL_CLUSTER_ONLY**.

The frozen 330-condition CPU experiment launched on 2026-10-12 at 01:42:33
Asia/Shanghai (2026-10-11 17:42:33 UTC). It completed on 2026-10-12 at 01:43:41
Asia/Shanghai, exit code zero. Supervisor wall time was 67.78 seconds; scientific
evaluation time was 66.94 seconds. All 330 expected conditions completed.

All five original Round 6 width-64, two-layer histogram checkpoints matched
their expected SHA256, architecture, 2,000 training steps and recorded quality
metadata. No training or checkpoint substitution occurred. All nine transferred
source files matched the frozen preparation. Four server numerical/software
tests passed; the original-model smoke completed all 11 conditions in 2.51
seconds, with an architecture replay maximum absolute gap of about 1.33e-15.

The earlier 167/330 dispatch snapshot is superseded by the retrieved completed
endpoint. Private operational receipts, checkpoint paths and failed connection
records are preserved locally and excluded from this public status record.

The scientific question is whether the reference-only local analytic bounds
remain informative after reducing old-prefix records. The primary budget is
n=64, m=32. Each method must achieve upper KL(Q||P) <=0.1 on at least two of
three fixed prefixes for at least four of five training seeds. The full protocol
also reports 25% and 75% record reductions and a near-full control.

At n=64,m=32 all methods have 0/5 passing models and 0/15 prefixes with upper
KL<=0.1. Agglomerative mean upper KL is 8.4763, versus actual KL 5.4366e-6;
all 15 bounds improve on the trivial reference cap, but only one is below 1.
Near-full n=64,m=63 agglomerative has 15/15 upper KL<=0.1 (14/15<=0.02).
This positive control reduces prefix records by only 1/64, not half.

Execution used float64, two CPU threads and a one-hour hard limit. These are
analytic-bound evaluations, not outward-rounded interval certificates. Exact
finite-tree KL is an independent validation measurement. No practical memory
or speed advantage was measured.

Start with [the reviewed results](../../RESULTS.md), then the completed
[snapshot summary](snapshots/20261011T174341Z/summary.json),
[record checks](snapshots/20261011T174341Z/verification.json) and
[330 condition metrics](snapshots/20261011T174341Z/condition_metrics.csv).
The local checker verifies frozen provenance, expected keys, recorded tree
sizes, both sequence inequalities, group/seed arithmetic and the gate. It does
not reconstruct omitted per-history trees, independently replay weights or
certify interval arithmetic. The frozen protocol's prospective status line
describes preparation time. The unchanged generated report title refers to
the original Round 6 models; this experiment is Round 9.

Read [the frozen protocol](../../experiments/PROTOCOL_ROUND9_LOCAL_CERT_V1.md)
and [runner](../../experiments/round9_original_models.py). Existing Round 6/7
results remain unchanged.
