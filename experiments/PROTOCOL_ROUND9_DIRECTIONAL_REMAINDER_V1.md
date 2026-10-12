# Round 9: one-case analytic directional remainder feasibility

The question is whether a computable second-derivative bound can turn the
previous accurate JVP into a small generation-KL upper estimate without using
actual compressed-model endpoint states in the bound. Reuse seed 11, first
n=64 prompt, m=32 positive cluster summary, T=4 and all 40 histories from the
completed diagnostic. No training, model/prefix selection, parameter sweep,
theta-grid fitting or oracle remainder substitution is permitted.

For each vector node propagate its exact P value x0, signed path tangent x1,
and an analytic norm bound B>=sup_[0,1] ||x''(theta)||. Linear maps preserve the
signed tangent; only second-order uncertainty is bounded by an operator norm.
LayerNorm uses the affine segment x0+[0,1]x1 plus the Taylor remainder ball,
exact GELU uses safe derivative bounds, and attention carries explicit affine
prefix masses, a centered numerator and a positive denominator enclosure.
Subsequent suffix KV inherit these same path jets. The method uses the model,
P trajectories and the fixed summary; it never reads Q endpoints to construct
the bound. Preserve the original absolute token positions and all vocabulary
symbols.

The endpoint centered-logit norm is bounded by ||z1||+B_z/2. Conditional KL is
at most its square divided by four (temperature one). Intersect with the
existing Round 9 conditional upper bound and reference universal cap. Use the
existing valid full-tree maximum-path/entropy-transfer aggregation; do not
average under P as a substitute for Q. The frozen feasibility target is
sequence KL upper<=0.1 on this one case. This is not the old five-model gate.

Before the scientific calculation, check the weighted-attention quotient,
LayerNorm/GELU jets and full-unroll value/tangent parity using small software
fixtures. During the run compare values and tangents against the frozen
diagnostic, with tolerance 1e-8. Actual Q changes and remainders are independent
validation only. A recorded endpoint inequality violation >1e-9 invalidates
the implementation and is preserved separately. If an analytic radius grows
too large for finite numerical evaluation, explicitly retain infinity and the
first failing stage; the old bound remains the safe fallback. Do not silently
clip a second-derivative bound into a successful estimate.

Primary outputs: per-history tangent norm, second-derivative upper, Taylor
remainder upper, actual measured remainder only for validation, conditional
and sequence risk estimates, and first-stage growth diagnostics. Retain source,
input and checkpoint hashes. Histories are dependent nodes of one case; no
confidence interval or generalization claim is appropriate.

This first implementation evaluates real-arithmetic inequalities in float64.
It is explicitly **not outward-rounded interval certification**. A numerical
pass would justify implementing validated arithmetic; a fail is a limitation
of this enclosure method, not evidence that every directional method fails.
The previous JVP result remains unchanged regardless of this outcome.

From the standalone repository root, with a fresh output directory:

```text
python -B -m unittest discover -s experiments -p test_round9_directional_remainder.py -v
python -B experiments/round9_directional_remainder.py --config experiments/configs/round9_directional_remainder_v1.json --checkpoint-root PATH_TO_ORIGINAL_MODELS --run-dir runs/NEW_REMAINDER_RUN
```
