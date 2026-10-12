# Round 14B: final suffix-write mechanism test and one-shot control

This protocol is frozen before scientific training. Configuration:
`configs/round14b_suffix_write_v1.json`. The source/config SHA256 manifest copied
into each new run is the authoritative frozen version. Smoke outputs are separate.
Round 14's negative result and all earlier evidence remain unchanged.

## Question and task

Does the retrospective full event-Fisher diagnostic explain generation KL better
than its diagonal on a trained retrieval task where compression touches newly
written values? Separately, does actual one-shot GS thinning remain useful on
these models under the Round 7 intervention, without claiming a nonvacuous bound?

Five new two-layer, single-head, pre-LayerNorm causal decoders use width 128,
GELU MLP width 512, learned absolute positions, vocabulary 12, context 184, no
dropout. Seeds are 11, 23, 37, 51, 71. Train exactly 5,000 AdamW steps, batch 64,
learning rate .001, warmup 200, cosine floor .05, weight decay .01, clip 1.
There is no selection of a best checkpoint or rescue training.

Keys are 0..3, values 4..7, START=8, WRITE=9, READ=10, FILL=11.
The eight-token prompt is START, a fresh uniform permutation of four keys, and
three FILL tokens. It contains no values. Each of 16 eleven-token suffix blocks is
`WRITE ka kb va vb FILL FILL READ query answer FILL`.
Keys ka,kb are uniform distinct ordered keys; va,vb are independent uniform
values sampled with replacement; query selects one of these keys uniformly.
The correct answer is its latest valid write. Two filler tokens separate the
write from READ. This is a deliberately simple synthetic task, not MQAR reproduction.
Every suffix token receives next-token cross entropy during training. Inputs are
prompt plus suffix[:-1]; the final prompt logit predicts the first suffix token,
so exactly 176 labels are scored and the first seven logits are ignored. Random
values/keys are scored by NLL and probability diagnostics, not 90% top-1.

Evaluation uses the model's complete unmasked softmax, including random-token
positions and invalid outputs. At fixed block positions, an invalid WRITE,
non-distinct/invalid keys, or invalid value pair does not update the memory map.
The query must be in the CURRENT valid write pair, matching the training support.
Invalid READ/query/answer contributes failure, rather than disappearing from the
denominator. Token histories determine truth independently of cache interventions.

## Continuous intervention

After consuming vb at suffix offset 4 in each block, compress the two adjacent
va/vb records at every layer. Plus retains the first original atom at weight 2;
minus retains the second at weight 2. Choice never depends on token ID.
This is a post-consumption cache edit: vb's forward computation uses the
uncompressed appended cache, including at higher layers. Its already computed
next-token logits stay unchanged. Subsequent token processing uses the edited
cache. The local attention diagnostic compares the two measures at vb's query;
it is a hypothetical response difference, not vb's realized hidden-state error.
All other cache records remain intact. Each compressed model computes its own
new queries, values and suffix cache; it never receives exact-stream suffix KV.
Events are fixed by the task's scheduled value slots, including on malformed
generated blocks, so event counts are identical across policies and phases.

Balanced Block uses eight plus then eight minus events. Alternating uses
plus/minus alternation. Both antipodes are evaluated, 16 Q-law paths per phase,
two heldout prompts. Each seed/policy averages prompts and both phases equally.
At theta=1, 32 eligible value records become 16; total final cache is 168 versus
184 records (8.70% reduction), not half the entire cache. Mass is preserved.
For these storage counts, the final sampled token is appended in an unscored
terminal forward call; its next-token logits do not enter the length-T KL.
Each two-token event increases total logical mass by TWO, not one: a stream of
unit pairs has mass N+2j before other ordinary appended tokens are counted.

The positive-measure smooth path uses masses (1-theta,1-theta,2theta) on the
two original atoms and selected original atom. Theta=.25 is a separate local
diagnostic with extra atoms, not the endpoint's physical budget. Both amplitudes
generate their own Q paths and report chain-rule KL. Only theta=1 can pass gates.

Compute conditional categorical KL(Q||P) exactly at every token on Q-sampled
histories, then average its cumulative sum. Thus full sequence KL is estimated
by Monte Carlo, not enumerated exactly. Save paths, per-token KL, full/diagonal
Fisher and event Grams at horizons 11,44,88,176. Full event JVPs differentiate
the entire causal suffix unroll at zero compression, including later KV/query
response. Fisher is averaged over those endpoint Q histories: a retrospective
fixed-history diagnostic, not P-law sequence Fisher, a heldout prediction,
certificate, or the Taylor coefficient of sequence KL including law change.

## Cache-only memory necessity control

Use 32 frozen legal heldout episodes per seed and exactly the same tokens,
READ queries, truth labels and positions in the two streams. Compare full KV
with deletion of both just-written value records in every layer immediately
after vb has been consumed. All other records remain; all future layer states
and KV are recomputed within each stream. Score teacher-forced READ-answer
argmax on all 512 answer positions. Later KV may relay value information; if
they defeat this ablation, the memory-necessity gate fails. Inputs are never
randomized and compared to obsolete labels.

## One-shot Round 7 control

For each of the same two heldout legal episodes, prefill prompt plus seven
full blocks and the eighth WRITE through vb: n=90 exact records. This cutoff
precedes the corresponding delayed READ, so old-prefix values still matter.
Apply actual `gs_thin_indices` with the
softmax KV Gram, or uniform subset, once independently per layer, m=45,
retained weights n/m=2. Use four summary seeds 101,103,107,109. The Gram uses
each exact prefix layer's maximum value norm as a recorded data-dependent scale;
this does not establish the model-wide constants of Round 7's theorem.
Compare also full cache. Continue the six remaining tokens of the eighth block
and eight further blocks for T=94, with each model appending its own unit-weight suffix KV and NO further
compression. The first next-token logits were already computed by exact prefill
and are shared; subsequent logits respond to each cache. Sample 16 paths from Q
for KL(Q||P) and independently 16 from P for KL(P||Q), at T=5,11,44,94.
T=5 includes the first delayed READ answer after the frozen write.
Never average models' logits as a mixture. Report summary-seed dispersion and
both directions per model seed. This arm matches Round 7's one-shot intervention;
the continuous arm does not inherit that theorem. No certificate or space-rate
claim is inferred from one budget and one prefix length.

## Frozen decision and statistical unit

The independent unit is a trained model seed. Prompts, paths, antipodes, summary
seeds and token positions are repeated measurements, not independent samples.
Report all five seeds and equal-seed means, ranges and Monte Carlo standard
errors within cells. No selective qualified-seed primary aggregate.

For each seed the continuous endpoint must meet ALL of:

1. Task: heldout teacher-forced READ-answer top-1 >=.90, unmasked P-generation
   answer accuracy >=.90, syntax-valid and query-valid rates each >=.95.
2. Memory necessity: full-minus-cache-deletion answer accuracy >=.20.
3. Matched intervention: physical theta=1 records, total mass, event counts and
   both balanced antipodes match; the ratio of the larger to smaller mean local
   attention error squared of the two policies is <=1.25 (zero/zero defined 1,
   positive/zero infinite). Means include both layers, prompts, phases and events.
4. Material risk: at least one policy has sequence KL >=.01 nat at T=176.
5. Explanation: for BOTH policies at T=176, full Fisher relative error
   `abs(full-KL)/max(KL,1e-12)` <=.25, and its absolute error is strictly less
   than diagonal Fisher's error.

The strongest transfer verdict requires the SAME >=4/5 seeds to satisfy 1–5.
Gates 1–4 are prerequisites; a Fisher pass alone cannot establish transfer.
If fewer than four meet prerequisites, label TASK_OR_INTERVENTION_UNQUALIFIED.
If prerequisites pass but joint explanation fails, label FISHER_TRANSFER_FAIL.
Otherwise label FISHER_TRANSFER_SUPPORTED_IN_THIS_TOY. Gate 4 is mandatory.
One-shot results are a separate descriptive control, never a rescue gate.
The one-shot arm ends with 139 records versus continuous compression's 168;
these arms have different budgets and cannot isolate compression timing at a
matched total budget. Gate 3 compares Block against Alternating only.

Execution errors/timeouts are INCOMPLETE, not scientific failure or success.
The supervisor writes a terminal closure report on success OR failure. The
two-hour total cap includes training, all evaluations and analysis. No new seed,
threshold relaxation, parameter sweep or Round 16 follows this run.

## Reproduction

From the standalone repository root, with Python, NumPy and PyTorch installed:

```text
python -B -m unittest discover -s experiments -p "test_round14b_*.py" -v
python -B experiments/round14b_train.py --config experiments/configs/round14b_suffix_write_v1.json --run-dir runs/NEW/training --device cuda:0
python -B experiments/round14b_evaluate.py --config experiments/configs/round14b_suffix_write_v1.json --training-dir runs/NEW/training --run-dir runs/NEW/evaluation
```

Checkpoint bytes and operational receipts remain private. Config, source hashes,
checkpoint hashes, per-seed metrics, paths and terminal scientific report are
the reproducibility record. This run does not establish NAACL acceptance or
novelty, actual inference speedup, optimal memory complexity, or a useful bound.
