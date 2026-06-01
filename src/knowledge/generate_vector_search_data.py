"""
src/knowledge/generate_vector_search_data.py
=============================================
Generate training data for the *vector_search* cross-encoder reranker.

vector_search is a GENERAL semantic search over every chunk type
(dosage, warnings, adverse reactions, interactions, indications, ...),
so the training set is sampled to be as diverse as possible across
drugs and section types -- unlike the interaction-only reranker.

For each sampled chunk we ask Grok for:
    good_query : a question the chunk clearly answers   (label 1.0)
    bad_query  : a hard negative that sounds similar but
                 is NOT answered by the chunk            (label 0.0)

We additionally mine cross-chunk hard negatives: a good_query paired
with a *different* chunk (label 0.0) to teach the reranker to
discriminate between superficially similar passages.

Output: data/vector_search_pairs.json   (query / chunk / label)
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

from knowledge.chroma_config import get_chroma_db_path, get_collection_name

DB_PATH         = get_chroma_db_path()
COLLECTION_NAME = get_collection_name()
OUTPUT_PATH     = os.path.join(ROOT_DIR, "data", "vector_search_pairs.json")

# How many chunks to turn into training examples. Each chunk yields
# ~3 labeled pairs (1 positive, 1 same-chunk negative, 1 cross negative).
MAX_CHUNKS      = int(os.getenv("VS_MAX_CHUNKS", "150"))
GROK_MODEL      = os.getenv("XAI_TEXT_MODEL", "grok-3")
MAX_CHUNK_CHARS = 1500          # keep passages within cross-encoder limits
MIN_CHUNK_CHARS = 120           # skip tiny / useless chunks

client = OpenAI(
    api_key=os.environ.get("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)


def _load_chunks() -> list[dict]:
    """Read every chunk from drug_db/drugs."""
    col = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
    data = col.get(include=["documents", "metadatas"])
    chunks = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        if doc and len(doc) >= MIN_CHUNK_CHARS:
            chunks.append({"text": doc, "metadata": meta or {}})
    return chunks


def _stratified_sample(chunks: list[dict], k: int) -> list[dict]:
    """
    Sample up to k chunks spread evenly across (drug_name, section_type)
    buckets so the reranker sees the full variety of content.
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for c in chunks:
        key = (
            c["metadata"].get("drug_name", "?"),
            c["metadata"].get("section_type", "?"),
        )
        buckets[key].append(c)

    bucket_lists = list(buckets.values())
    for b in bucket_lists:
        random.shuffle(b)

    selected: list[dict] = []
    # round-robin across buckets until we hit k
    idx = 0
    while len(selected) < k and any(bucket_lists):
        bucket_lists = [b for b in bucket_lists if b]
        if not bucket_lists:
            break
        b = bucket_lists[idx % len(bucket_lists)]
        selected.append(b.pop())
        idx += 1
    return selected[:k]


def _truncate(text: str) -> str:
    return text[:MAX_CHUNK_CHARS]


def _grok_queries(chunk_text: str) -> dict | None:
    """Ask Grok for one good and one bad (hard-negative) query."""
    system_prompt = (
        "You are an expert medical data annotator generating training data "
        "for a Cross-Encoder reranker used in a general medical search engine. "
        "Respond ONLY with a raw JSON object with exactly two keys: "
        "'good_query' and 'bad_query'. No markdown, no code fences."
    )
    user_prompt = (
        f'Read this medical label text chunk:\n"""{chunk_text}"""\n\n'
        "1. good_query: a natural, specific question a patient or clinician "
        "might type that THIS chunk directly and fully answers.\n"
        "2. bad_query: a tricky question that reuses similar medical terms "
        "from the chunk but that the chunk does NOT actually answer."
    )
    try:
        resp = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        # be tolerant of accidental code fences
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[raw.find("{"):]
        out = json.loads(raw)
        if "good_query" in out and "bad_query" in out:
            return out
    except Exception as exc:  # noqa: BLE001
        print(f"   ! Grok error: {str(exc)[:100]}")
    return None


def main() -> None:
    print(f"Loading chunks from {DB_PATH} ...")
    all_chunks = _load_chunks()
    print(f"  {len(all_chunks)} usable chunks found.")
    if not all_chunks:
        raise SystemExit(
            "No chunks available. Populate the label DB first, e.g.\n"
            "  PYTHONPATH=src python src/knowledge/ingest_data.py\n"
            "and ensure CHROMA_DB_PATH points at your populated medlabel_db."
        )

    sample = _stratified_sample(all_chunks, MAX_CHUNKS)
    print(f"Sampling {len(sample)} diverse chunks for query generation "
          f"(model={GROK_MODEL}).\n")

    pairs: list[dict] = []
    for i, chunk in enumerate(sample, 1):
        drug = chunk["metadata"].get("drug_name", "?")
        sec  = chunk["metadata"].get("section_type", "?")
        print(f"[{i}/{len(sample)}] {drug} / {sec}")

        qs = _grok_queries(_truncate(chunk["text"]))
        if not qs:
            continue

        passage = _truncate(chunk["text"])

        # positive
        pairs.append({"query": qs["good_query"], "chunk": passage, "label": 1.0})
        # same-chunk hard negative
        pairs.append({"query": qs["bad_query"],  "chunk": passage, "label": 0.0})
        # cross-chunk hard negative: good query vs an unrelated chunk
        other = random.choice(sample)
        if other is not chunk:
            pairs.append({
                "query": qs["good_query"],
                "chunk": _truncate(other["text"]),
                "label": 0.0,
            })

        time.sleep(0.3)  # gentle rate limiting

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)

    n_pos = sum(1 for p in pairs if p["label"] == 1.0)
    n_neg = sum(1 for p in pairs if p["label"] == 0.0)
    print(f"\nDone. {len(pairs)} pairs  ({n_pos} positive / {n_neg} negative)")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
