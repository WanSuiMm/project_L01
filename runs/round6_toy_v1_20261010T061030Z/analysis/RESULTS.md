# Round 6 analysis

Analysis inputs are read for run `round6_toy_v1_20261010T061030Z`; the source summaries are not copied into this derived directory.
The report and six figures in this directory are derived summaries; no experiment was launched here.

## Experiment 1: exact analytic toy

The Exp1 summary contains 160 exact `(n, T, epsilon)` rows; all recorded analytic gates pass: **True**.
The plotted conditional logits come from weighted KV numerator/denominator, $a_t=\epsilon n/(n+t)$; KL is the exact finite sum of Bernoulli conditional KL terms.
The saturation coordinate is $nT/(n+T)$. The plotted lower and upper slopes are $\sigma'(1)/6$ and $1/8$ after normalization by $\epsilon^2$.
This is a deterministic toy construction. It is not a learned-model result, an empirical novelty claim, or a minimax space lower bound.

## Experiment 2: trained decoder results

Exp2 status: **not_supplied**. Exp2 directory not found.
No learned-model scientific claim is made from absent, incomplete, or smoke-only output.

## Figures

- `kl_horizon_scaling.png` / `.pdf`: exact Exp1 cumulative KL across horizon and saturation scale.
- `precision_scaling.png` / `.pdf`: exact Exp1 epsilon-squared normalization and theorem envelope.
- `multilayer_autoregressive_fidelity.png` / `.pdf`: Exp2 seed-level KL intervals and actual Q-cache record/byte costs, or an explicit no-data panel.

Machine-specific paths, checkpoints, and raw per-prompt rows are not copied into this derived report.
