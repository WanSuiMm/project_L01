# Frozen public input attribution

The 24 prefixes in `wikitext_v1_20261009T190129Z/prompts.jsonl` derive from
WikiText-2 raw validation, distributed as
[Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext).
The dataset card lists CC-BY-SA-3.0 and GFDL. Those terms apply separately to
the corpus-derived text; no alternative repository-wide license is asserted.

Each row records its article title, dataset source, deterministic article ID and
calibration/heldout split. `input_manifest.json` pins the dataset revision,
source parquet hash and prompt hash. One article appears in one split only.
See [ATTRIBUTION.md](ATTRIBUTION.md) for article/history links and license links.

Transformations: concatenate non-heading article paragraphs, take the first
1800 characters, and prepend a continuation instruction. At inference, shorten
only user content to fit the 384-token chat-template limit. The runner records
raw/used content token counts for every case.

The parquet file and model weights are not published. Frozen text prefixes and
their provenance suffice to replay the recorded input. The preparer is provided
for future input acquisition; use the published frozen file for exact replay.
