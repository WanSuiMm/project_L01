# Round 12: balanced temporal ordering and actual prefix bias

Frozen before the scientific launch, 2026-10-12. This is a finite trained-model
mechanism experiment, without retraining, asymptotic claims or certification.
Use `configs/round12_balanced_order_v1.json` and a fresh output directory.

## Question and intervention

Does the large coherent-versus-alternating gap from Round 11 persist when
both sign schedules have exactly the same positive and negative event counts?
How does that gap relate to **actual bias at every prediction prefix**, rather
than a terminal sign count or a terminal feature statistic?

Reuse the original five qualified Round 6 histogram models (seeds 11,23,37,51,71;
width 64, two layers, 2,000 training steps), checked against the frozen hashes
and validation metadata. Use the same four n=64 prompts and prompt seed as
Round 11. No checkpoint, tokenizer, training distribution or weight is changed.

At input steps 2,4,...,128, replace the previous/current generated-token pair
with one original KV record of weight 2, before the current attention calculation.
Plus retains the higher token ID (ties retain the earlier record); minus retains
the other member. The original n-token prefix and START record remain exact.
Subsequent KV states are computed normally using each compressed model's own
states and generated history. Both layers use the same externally fixed sign.

Exactly 64 events occur in T=129 predictions:

| Policy | Phase plus | Phase minus |
|---|---|---|
| Balanced Block | 32 plus followed by 32 minus | its antipode |
| Alternating | 32 repetitions of plus,minus | its antipode |

Each phase has 64 Q-law paths. Uniform random numbers are shared across
policies **and phases** for each model/prompt/path. Models can generate different
histories under the same uniforms. Phase averaging targets mean fixed-schedule
risk, not KL of the policy mixture. The independent scientific unit is the five
training seeds, with equal weighting of phases and prompts within each seed.

## Timeline and primary estimand

At prediction t=step+1, consume START if step=0, otherwise Y_step, perform
any eligible compression, measure the caches and the conditional distributions,
then sample Y_t from the full three-token Softmax of Q. No top-k/top-p truncation.
The last sampled Y_129 is recorded but not inserted into the measured final cache.
The final cache represents the original prefix, START, and Y_1,...,Y_128.
Its logical mass is 193 and its physical record count is 129. This halves the
128 generated input-token records; it is **not** a whole-cache half-budget test.

For each Q history, run an independent, exact full-cache P replay of that same
discrete history. At every t, evaluate the full categorical conditional
KL(Q_t || P_t) and sum it. Averaging these sums under Q estimates the full
sequence KL(Q_1:T || P_1:T) by the chain rule. This integrates the next-token KL
exactly; the expectation over histories is Monte Carlo, not an exhaustive-tree
calculation. Retain every prefix, not only milestones t=17,33,65,129.

## Actual bias measurements

At each prediction prefix, logical mass M=n+step+1:

1. **Token composition:** b_tok is the weighted histogram of physically retained
   tokens minus the uncompressed histogram of the same Q input history. Record
   its raw vector, squared norm, and normalized vector b_tok/M and squared norm.
   Independently reconstruct the weighted physical histogram to check the
   cumulative event update. Summed token mass discrepancy must be zero.
2. **Direct local KV perturbations:** for every layer, accumulate the observed
   event vector 2[K_selected,V_selected]-[K_previous,V_previous]-[K_current,V_current]
   evaluated on Q's actual evolving states. Record its raw squared norm and
   normalized squared norm. This sum alone omits changed later representations.
3. **Actual cache moments:** per layer, subtract full P replay's weighted K/V sum
   from Q compressed cache's weighted K/V sum on the **same Q discrete history**.
   Divide by M and record per-layer and total squared norms. This includes
   compression and representation drift. Record the normalized squared norm
   of actual cache discrepancy minus the direct perturbation sum separately.
   These unweighted-by-query moments are interpretable diagnostics, not a
   sufficient statistic for attention or generation risk.
4. **Local distortion:** measure event-time pre/post-compression attention
   squared error and denominator relative error, by layer and phase. Report
   the event/layer average alongside sequence risk; equal counts alone do not
   ensure equal local distortions.

Always average squared path magnitudes; never square an antipodal phase mean.
Signed token means are retained per phase. Save the conditional/cumulative KL
and scalar bias trajectories jointly for every path in compressed NPZ files,
with inputs, sampled outputs, uniforms, sign schedules and metric names.
This enables path-level joint checks without conflating different Q laws.

## Coverage, checks and interpretation

Expected formal coverage: 10,320 prefix rows; 10,240 per-phase/layer event rows;
5,120 terminal path rows; 40 joint path trajectory files. Analysis checks unique
prefix and terminal keys, full coverage, balance and finite values. Software
checks test the hand-computable physical histogram, original-model KL chain
rule on a tiny enumerable tree, and the layer-0 identity: its actual raw cache
moment discrepancy equals the direct event-vector sum, since layer-0 K/V does
not depend on earlier attention. At every prefix assert physical counts,
logical mass, exact original prefix/START records and histogram reconstruction.

Report block>alternating terminal KL in at least 4/5 seeds as an ordering
screen, and the event/layer-pooled local-error max/min<=1.25 separately as a
comparability screen. Also retain per-layer local means to expose mismatches
hidden by pooling; a pooled pass alone does not establish layer-wise matching.
Report seed-level effect sizes, phase details, terminal token/cache bias and
the sum over prefix squared-bias magnitudes. No significance test uses paths
or prefix rows as independent model replicates. Monte Carlo uncertainty is
reported separately from variation across training seeds.

Zero terminal sign sum does **not** match terminal token/KV bias. A surviving
gap does not establish an effect beyond bias; a disappearing gap does not
prove bias is the sole cause. Relate KL to the whole measured bias trajectory,
including the block switch at event 32; keep discrepant cases and negative
results. Neither policy is expected to achieve an asymptotic bounded-KL law.

The next strengthening stage, after reading this result, is a same-horizon,
same-KL-direction Fisher/full-unroll comparison with the actual primary risk.
This release does not launch that stage or tune it using the current outcome.
Further theory is justified only by a remaining explanatory gap.

## Execution and reading order

CPU, float64, two Torch threads, frozen 45-minute wall-time cap. First run a
tiny seed-11 smoke (T=9, one prefix, four paths/phase) and targeted software
tests, then deploy the frozen five-model protocol. Operational machine/PID
receipts stay private. Launch does not create an ongoing monitor.

Read `summary.json`, `REPORT.md`, `identity_checks.json`, and the prefix CSV
first. The larger `paths/*.npz` files are secondary joint evidence. A completed
process is not a positive scientific conclusion. Existing Round 6–11 evidence
and certificate failures remain unchanged.

From the standalone repository root:

```text
python -B -m unittest discover -s experiments -p test_round12_balanced_order.py -v
python -B experiments/round12_balanced_order.py --config experiments/configs/round12_balanced_order_v1.json --checkpoint-root PATH_TO_ORIGINAL_MODELS --run-dir runs/NEW_ROUND12_RUN
```

The runner freezes sources, configuration, model hashes and prompts in the new
run directory, and automatically generates analysis after successful coverage.
