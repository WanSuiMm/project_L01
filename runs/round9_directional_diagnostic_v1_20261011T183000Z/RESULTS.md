# Round 9 one-case slack and JVP diagnostic

Status: ORACLE_SLACK_DIAGNOSTIC_NOT_A_CERTIFICATE; execution COMPLETE.

Frozen seed 11, prefix histogram-n64-0000, n=64,m=32,T=4, positive agglomerative means.
One original trained model; no retraining. All 40 prediction histories and 81 leaves retained.

Replayed Round 9 upper KL(Q||P): 36.623076; exact sequence KL: 3.45821894e-06.

## Slack decomposition

| Diagnostic | Median over histories | Maximum |
|---|---:|---:|
| Hidden propagation: radius / actual norm | 166.42126 | 295.065523 |
| Final LN: chord bound at actual norm / actual LN delta | 1.12011299 | 1.22030266 |
| Output direction: head norm bound / actual centered logits | 5.74785436 | 15.9518345 |
| Actual LN delta fraction in centered-head row space | 0.237105365 | 0.299376782 |
| Softmax: exact-logit quadratic bound / actual conditional KL | 1.99251496 | 11.3839873 |
| Composed Round 9 logit norm bound / actual logit norm | 2527.13293 | 7963.97305 |
| JVP vector remainder / max(endpoint,JVP,1e-10) | 0.0185004268 | 0.0636924009 |
| JVP versus endpoint cosine | 0.99992304 | 0.999994184 |
| Hidden JVP scaled remainder | 0.0149438312 | 0.0265133944 |

Histories are dependent tree nodes, not independent samples. Level-wise exact P-weighted means are in summary.json.
Raw norms and vectors are secondary evidence in histories.jsonl; read this aggregate first.

## Oracle substitution diagnostic

The following use actual Q and Q history masses. They are not reference-only certificates.

| Quantity summed over Q histories | Sequence sum |
|---|---:|
| exact_conditional_KL_QP | 3.45821894e-06 |
| round9_conditional_upper_QP | 21.6055753 |
| oracle_actual_hidden_norm_KL_upper | 0.00015628607 |
| oracle_actual_LN_norm_KL_upper | 0.000119210065 |
| oracle_actual_logit_KL_upper | 3.47005889e-06 |
| true_logit_quadratic_KL_bound | 9.71329707e-06 |

Frozen first-order screen: PASS; worst scaled logit remainder 0.0636924, threshold 0.25.
Passing supports a local first-order description on this interpolation, not a bound on its nonlinear remainder or a project-wide mechanism.
A small endpoint/JVP response alone does not identify output-null alignment as the cause.

## Verification and scope

checks.json records endpoint identity, forward-JVP/finite-difference agreement,
original checkpoint/prefix/source identity, Round 9 replay and tree mass checks.
All quantities are float64 oracle diagnostics, not outward-rounded certificates.
No second-derivative supremum, long-horizon guarantee, general NLP claim or systems benefit was evaluated.
