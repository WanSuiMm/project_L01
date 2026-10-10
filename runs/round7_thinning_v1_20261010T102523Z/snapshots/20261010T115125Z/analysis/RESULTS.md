# Round 7 analysis

Combined status: **COMPLETE**. Trained-row status: **COMPLETE**; the configured grid expects 70,560 rows, 70,560 were found, and 70,560 passed validation as direction-paired rows.
Configured models: L=2: 5 seeds, L=4: 5 seeds; n=[16, 64, 128]; T=[8, 32, 128]; budget fractions=[0.125, 0.25, 0.5]; prompts=8; compression seeds=8; paths per direction=16.
Expected by method: full=1,440, gs_thin=34,560, uniform_subset=34,560. Missing keys=0; unexpected valid keys=0; duplicate keys=0; row checks: none.
Primary trained estimates use sequence KL from both directions. Within each training seed, compression seeds are averaged inside each prompt and prompts receive equal weight; percentile bootstrap intervals resample training seeds only. Valid per-row KL ranges: Q→P 0–10.24, P→Q 0–9.077. Zero-mean aggregate KL groups: 36.
Frozen checkpoint-quality gate: **PASS**; 10/10 checkpoints pass bayes_nll_gap ≤ 0.03.
GS-Thin forward small-risk certificate: 0/17280 Q→P rows satisfy the audited V<1/4 condition; observed Q→P KL range is reported separately as [1.1381896875300076e-05, 10.239222379940774]. Full identity rows are excluded. An uncertified condition is not an observed-KL violation, and empirical MMD is not the theorem certificate.
Paired GS-Thin minus uniform-subset comparisons: 108 cells; 108 have seed-bootstrap intervals. The observed difference range is -0.6114–0.2295; positive differences mean GS-Thin has larger KL. These are descriptive endpoints, not theorem verification.

## Analytic endpoint

Analytic input status: **ANALYTIC_GATES_PASS**; all analytic gates pass: **True**; aggregate rows=2889, hard equal-weight rows=90, balanced-uniform rows=90, exact witness rows=90, exact estimates parsed=504, envelope rows=180.
Exact minima/maxima over feasible equal-weight summaries, exact expected uniform-subset risks, and one exact minimum-gap witness are kept as separate endpoint classes. The witness is not a minimax risk. Longer-horizon curvature bounds remain envelopes, not measured KL values.

## Interpretation and figures

Round 7's proof claim is limited to the stated sub-Gaussian summary interface and the specified equal-weight subset thinning class. The trained decoder grid is an empirical transfer check; it does not establish the theorem's uniform assumptions or a general minimax lower bound. Optional positive-cluster results remain diagnostic and are not merged into the equal-weight theorem class.
The budget plot shows KL/h against retained old-prefix records m. Its m^-1 and m^-2 guides are visual references only; no slope is forced or fitted. Zero-valued groups are omitted from log axes without adding an epsilon.

- budget_scaling.png / .pdf: sequence KL/h against actual m, split by prefix length and KL direction; reference exponents are visual only.
- horizon_scaling.png / .pdf: sequence KL against T at fixed m/n=0.25, split by prefix length and KL direction.
- multilayer_fidelity.png / .pdf: prefix MMD², upper-layer suffix value drift, and full-identity logit gap across model depths.

The raw JSONL, checkpoints, and private launch records are not copied into this analysis directory.
