# 7B pilot complete: no large Block/Alternating contrast observed

**Execution COMPLETE; no stable large ordering contrast on this fixed pilot.**
All 144 trajectories on Qwen2.5-7B-Instruct completed at 12:49:37 UTC
(20:49 China time) on 2026-10-12. The persistent job ran 516.50 seconds
(8 minutes 36 seconds). No new training, model download or rescue sweep occurred.

## Primary all-prompt result

Average both phases and two replicates within each of the twelve prompts, then
give every prompt equal weight. The primary KL is the Q-law Monte Carlo mean of
the sum of exact full categorical conditional KL(Q_t||P_t), T=130, temperature 1.

| Policy | Mean sequence KL(Q||P), nats | Mean direct local attention error squared |
|---|---:|---:|
| Balanced Block | 181.3447 | 31.9788 |
| Alternating | 185.2281 | 34.8241 |
| Balanced Random | 189.4147 | 35.5113 |

Block/Alternating aggregate KL ratio is **0.9790**; local-error max/min ratio is
1.0890. Block has greater KL in 6/12 prompts and lower KL in 6/12. The mean paired
KL difference is -3.8834 nats; the frozen paired-prompt bootstrap interval is
[-27.9022, 17.1016]. It spans zero and describes variation across these fixed
templates within one checkpoint, not uncertainty across pretrained models or
a randomly sampled task population.

## Prespecified local-comparability tags

Eleven of twelve prompts meet the pooled local-error max/min <=1.25 tag.
`reason_en_01` fails (1.4641); it remains in all primary aggregates above.
Among the eleven comparable prompts, Block/Alternating KL means are
182.4765/178.3918 (ratio 1.0229), with Block greater in 6/11. Their per-prompt
Block/Alternating ratios range from 0.7300 to 1.2889. **No comparable prompt has
a twofold contrast in either direction.** The protocol's descriptive count of
Block/Alternating >=2 jointly with local comparability is 0/12. This is a
negative effect-size screen, not a failed theorem or a universal impossibility.

| Prompt | Block KL | Alternating KL | B/A KL | Local max/min | Comparable |
|---|---:|---:|---:|---:|---|
| instruction_en_01 | 229.059 | 201.009 | 1.140 | 1.037 | yes |
| instruction_en_02 | 197.887 | 247.087 | 0.801 | 1.156 | yes |
| instruction_zh_03 | 151.164 | 207.062 | 0.730 | 1.155 | yes |
| instruction_zh_04 | 139.120 | 120.564 | 1.154 | 1.087 | yes |
| reason_en_01 | 168.896 | 260.427 | 0.649 | 1.464 | no |
| reason_en_02 | 224.000 | 211.739 | 1.058 | 1.228 | yes |
| reason_zh_03 | 151.659 | 153.717 | 0.987 | 1.044 | yes |
| reason_zh_04 | 136.820 | 106.154 | 1.289 | 1.159 | yes |
| story_en_01 | 227.640 | 198.761 | 1.145 | 1.098 | yes |
| story_en_02 | 253.914 | 208.353 | 1.219 | 1.117 | yes |
| story_zh_03 | 154.251 | 164.265 | 0.939 | 1.069 | yes |
| story_zh_04 | 141.726 | 143.597 | 0.987 | 1.028 | yes |

Comparable pooled magnitudes do not mean identical error vectors or small error.
The local metric sums squared pre-o_proj attention displacement over heads and
head dimensions, then averages layers/events. Signs select earlier/later KV
records and do not orient actual error vectors. Thus the pilot does not test an
identical signed-error intervention to the histogram construction.

## Uncertainty and EOS

Only two path replicates per phase are included. As a diagnostic, replicate 0
has mean Block-minus-Alternating KL -7.9874 nats, versus +0.2207 for replicate 1
(each averages phases within prompt). Neither is a new independent model
replication; no tight equivalence margin or absence of all smaller effects is
established. Balanced Random need not lie between the deterministic policies.

Two of 144 paths emit an EOS: `story_en_02`, Random phase +1 replicate 0 at
position 100, and Alternating phase +1 replicate 1 at position 82. Their post-EOS
conditional KL contributions are 69.0902 and 67.5851 nats. These paths remain
in the primary fixed-horizon metric. No Block path emits EOS. The measurement
therefore concerns the raw fixed-horizon model law, not stop-on-EOS completion
quality. No task accuracy was measured.

All three policies have high cumulative KL in this intervention. Similar policy
risks do not certify fidelity to the dense model. Only generated suffix pairs
are logically thinned, the prompt stays exact, and KVs remain allocated. The
logical record deletion fraction is 27.47--29.22% of the terminal total cache,
depending on prompt length; no physical memory or speed benefit is claimed.

## Interpretation and evidence

This pretrained-model pilot **does not reproduce the histogram model's 11--12x
contrast**, despite local comparability in 11/12 prompts. Earlier toy results
remain valid in their setting. The present evidence limits their direct
generalization to this checkpoint, these prompts, horizon and retention rule;
it does not prove the absence of temporal effects in other LLM interventions.
No automatic follow-up or additional experiment was launched.

Read the [frozen protocol](../../experiments/PROTOCOL_REAL_LLM_TEMPORAL_V1.md),
[small canonical summary](evaluation/summary.json), then the
[record checks](analysis/record_check.json). The 8 MB
[trajectory JSONL](evaluation/rows.jsonl) is secondary evidence; do not open it
first. The [completed manifest](evaluation/manifest.json) records the full grid
and runner/prompt identities. Private completion receipts are excluded from
publication.

Record-only verification confirms all 144 grid keys, twelve prompts, ten local
frozen source hashes, manifest/source agreement, balanced sign schedules,
130 conditional KLs and 64 events per path, finite nonnegative KL terms, exact
chain-sum arithmetic and zero discrepancy in independently recomputed prompt
aggregates. It does not replay the 7B weights or establish a statistical theorem.

From the repository root, reproduce the public record/source checks without
loading weights or rewriting frozen evidence:

```text
python -B scripts/verify_real_llm_temporal_v1_release.py --input-dir runs/real_llm_temporal_v1_20261012T123114Z
```

The source package's original descriptions and dispatch are historical snapshots.
Use this completed report for current status and the separate
[7B dependency file](../../experiments/requirements_real_llm_temporal_v1.txt)
for a fresh run. The original ZIP is retained locally; its SHA-256 is in FROZEN.json.
