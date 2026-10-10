# Round 6 analysis

Analysis inputs are read for run `round6_toy_v1_20261010T061030Z`; the source summaries are not copied into this derived directory.
The report and six figures in this directory are derived summaries; no experiment was launched here.

## Experiment 1: exact analytic toy

The Exp1 summary contains 160 exact `(n, T, epsilon)` rows; all recorded analytic gates pass: **True**.
The plotted conditional logits come from weighted KV numerator/denominator, $a_t=\epsilon n/(n+t)$; KL is the exact finite sum of Bernoulli conditional KL terms.
The saturation coordinate is $nT/(n+T)$. The plotted lower and upper slopes are $\sigma'(1)/6$ and $1/8$ after normalization by $\epsilon^2$.
This is a deterministic toy construction. It is not a learned-model result, an empirical novelty claim, or a minimax space lower bound.

## Experiment 2: trained decoder results

Exp2 status: **included_completed_formal**; 20 trained model records and 276 summary groups are retained.
The primary estimate is the training-seed mean of prompt means for the sum of conditional KL terms sampled under Q, an estimator of sequence $D_{KL}(Q\Vert P)$. Error bars use two-sided 95% t intervals across training seeds; prompt-cluster and continuation-MC errors are secondary.
Controls present: full, positive_cluster, uniform_subset, unweighted_subset. Every trained model remains in the aggregates, including seeds that fail task-quality gates.
Quality gates from `effective_config.json`: histogram Bayes NLL gap ≤ 0.03; recall answer top-1 ≥ 0.9. Qualified models: 13/20; unqualified: 7; unassessed: 0.
Actual Q prefix records per layer span 4 to 128; exact Q cache bytes per continuation span 12384 to 528384. These byte counts include suffix K/V and explicit weights; attention scratch remains an estimate.
There are 24 rho<1 precision-mixture groups. They are diagnostic only and store n+m physical prefix records; they are not m-record compressors. In the precision figure, rho is the intervention amplitude, not a certified epsilon; KL is plotted raw, and no monotonicity or quadratic KL law is assumed.
Observed-query moment errors and empirical epsilon values are diagnostics for the sampled queries only. They are not uniform-query theorem certificates and were not used to fit a constant.
Generated recall accuracy is reported with both denominators: conditional accuracy uses answers following valid generated queries only; all-answer-position accuracy counts an invalid generated query's answer position as incorrect. The frozen training-quality gate uses teacher-forced validation answer top-1 accuracy, not either generated metric.
Global norm audits are recorded for 20/20 models in log space (log C_model range 139.22 to 1870; log coarse uniform epsilon range 12.702 to 261.61). For applicable compressed groups, the derived log-KL upper bound is compared only with the model-specific bounded-logit sequence cap from the audit; it is no tighter in 840/840 group-seed cases with available values. Pinsker is separately flagged when its TV upper bound reaches one. No fitted C is reported.

## Figures

- `kl_horizon_scaling.png` / `.pdf`: exact Exp1 cumulative KL across horizon and saturation scale.
- `precision_scaling.png` / `.pdf`: exact Exp1 epsilon-squared normalization and theorem envelope.
- `multilayer_autoregressive_fidelity.png` / `.pdf`: Exp2 seed-level KL intervals and actual Q-cache record/byte costs, or an explicit no-data panel.

Machine-specific paths, checkpoints, and raw per-prompt rows are not copied into this derived report.
