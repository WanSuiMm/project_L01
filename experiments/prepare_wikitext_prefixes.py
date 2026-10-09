"""Freeze article-disjoint public prefixes; does not inspect model outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "L01-exp0-input-freezer/1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = response.read(4_000_001)
    if len(payload) > 4_000_000:
        raise ValueError("Input response exceeded the bounded 4 MB allowance")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "prompts.jsonl").exists():
        raise FileExistsError("Frozen inputs already exist; refusing regeneration")
    meta = json.loads(fetch("https://huggingface.co/api/datasets/Salesforce/wikitext"))
    revision = meta["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid dataset revision")
    relative = "wikitext-2-raw-v1/validation-00000-of-00001.parquet"
    source_url = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{revision}/{relative}"
    raw = fetch(source_url)
    source_path = output / "validation.parquet"
    source_path.write_bytes(raw)
    import pyarrow.parquet as pq
    records = pq.read_table(source_path, columns=["text"]).to_pydict()["text"]
    articles: list[dict] = []
    current: dict | None = None
    for line in records:
        text = line.strip()
        if re.fullmatch(r"= [^=]+ =", text):
            if current is not None:
                articles.append(current)
            current = {"title": text[2:-2].strip(), "paragraphs": []}
        elif current is not None and text and not text.startswith("="):
            current["paragraphs"].append(text)
    if current is not None:
        articles.append(current)
    eligible = []
    for article in articles:
        joined = "\n\n".join(article["paragraphs"])
        if len(joined) >= 1400:
            eligible.append({"title": article["title"], "passage": joined[:1800]})
    random.Random(20261010).shuffle(eligible)
    if len(eligible) < 24:
        raise RuntimeError(f"Need 24 distinct articles, found {len(eligible)}")
    prompts = []
    for index, article in enumerate(eligible[:24]):
        article_id = hashlib.sha256(article["title"].encode()).hexdigest()[:16]
        prompts.append({
            "id": f"wikitext2_validation_{article_id}",
            "split": "calibration" if index < 8 else "heldout",
            "article_title": article["title"],
            "text": "Continue the following passage naturally. Write only the continuation.\n\n" + article["passage"],
            "source": "Salesforce/wikitext:wikitext-2-raw-v1:validation",
        })
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompts).encode("utf-8")
    (output / "prompts.jsonl").write_bytes(encoded)
    manifest = {
        "dataset": "Salesforce/wikitext", "subset": "wikitext-2-raw-v1",
        "split": "validation", "revision": revision, "source_url": source_url,
        "source_sha256": hashlib.sha256(raw).hexdigest(), "source_bytes": len(raw),
        "prompts_sha256": hashlib.sha256(encoded).hexdigest(), "seed": 20261010,
        "eligible_articles": len(eligible), "calibration_prompts": 8, "heldout_prompts": 16,
        "article_ids": [row["id"] for row in prompts],
        "license": "CC-BY-SA-3.0 / GFDL (see original dataset card)",
    }
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
