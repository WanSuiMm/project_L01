# Observable gain analysis

**Status:** `NO_HELDOUT_SUPPORT`

Rows: 48 calibration and 96 held-out; independent prompts: 8 calibration and 16 held-out.

The primary comparison is held-out prompt-macro log-MSE for the candidate (current plus recent mean/max proxies) against current-proxy ridge. Each prompt contributes one averaged loss to the paired bootstrap.

- Current-proxy log-MSE: 0.104036; candidate log-MSE: 0.104228.
- Mean paired loss reduction: -0.000191404 (-0.18%); 95% prefix-bootstrap CI: [-0.000420286, 4.45716e-05].
- Held-out case-level Spearman: current 0.23936161830374858; candidate 0.24259359739555073; actual-current oracle 0.20128865979381444.
- Actual-current oracle log-MSE is reported as an informative comparison; it is not part of the decision gate.

Decision gate: `PILOT_SUPPORT` requires at least 10% mean primary MSE reduction and a strictly positive lower 95% bootstrap bound. Otherwise the result is `NO_HELDOUT_SUPPORT`; fewer than 4 calibration or 8 held-out prompts is `INSUFFICIENT_SUPPORT`.

Limitations: this is a short-horizon cumulative-KL prediction pilot, not a full-sequence TV certificate. It contains no systems speed or memory benchmark. The direct residual proxy is BPKV-style and is not a paper replication; recent-window features are heuristics, not a Gramian bound.

The target `future_cumulative_kl` excludes t0 and sums the configured 32 future positions.
