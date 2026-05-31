"""
Ingest data/drugs.json into ChromaDB (medlabel_db by default).

Usage (from repo root):
    PYTHONPATH=src python src/knowledge/ingest_data.py
    PYTHONPATH=src python src/knowledge/ingest_data.py --reset   # wipe collection first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Allow running as script from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledge.chroma_config import get_chroma_db_path, get_collection_name
import chromadb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "drugs.json")

# Only sub-chunk very long sections; JSON rows are usually already one section.
MAX_WORDS_BEFORE_CHUNK = 250
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= MAX_WORDS_BEFORE_CHUNK:
        return [text]
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        part = " ".join(words[i : i + chunk_size])
        if part.strip():
            chunks.append(part)
    return chunks


def ingest_json_to_chroma(*, reset: bool = False) -> int:
    db_path = get_chroma_db_path()
    coll_name = get_collection_name()
    os.makedirs(db_path, exist_ok=True)

    client = chromadb.PersistentClient(path=db_path)
    if reset:
        try:
            client.delete_collection(coll_name)
            print(f"Deleted existing collection '{coll_name}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(name=coll_name)

    print(f"DB: {db_path}  collection: {coll_name}")
    print("Loading BGE-M3...")
    model = SentenceTransformer("BAAI/bge-m3")

    print(f"Loading {JSON_PATH}...")
    with open(JSON_PATH, "r", encoding="utf-8") as file:
        all_drugs_data = json.load(file)

    all_ids: list[str] = []
    all_documents: list[str] = []
    all_embeddings: list[list[float]] = []
    all_metadatas: list[dict] = []

    print(f"Ingesting all {len(all_drugs_data)} JSON records...")
    for item in tqdm(all_drugs_data, desc="Processing"):
        full_text = (item.get("text") or "").strip()
        meta = item.get("metadata") or {}
        if not full_text:
            continue

        drug_name = (meta.get("drug_name") or "UNKNOWN").strip()
        section_type = meta.get("section_type", "unknown_section")
        data_source = meta.get("source", "unknown_source")
        rxcui = str(meta.get("rxcui") or "")
        brand_name = str(meta.get("brand_name") or "")

        text_chunks = chunk_text(full_text)
        base_id = item.get("id") or f"{drug_name}_{section_type}"

        for chunk_index, chunk in enumerate(text_chunks):
            unique_suffix = uuid.uuid4().hex[:6]
            chunk_id = f"{base_id}_{chunk_index}_{unique_suffix}".replace(" ", "_")

            vector = model.encode(chunk, normalize_embeddings=True).tolist()

            all_ids.append(chunk_id)
            all_documents.append(chunk)
            all_embeddings.append(vector)
            all_metadatas.append({
                "drug_name": drug_name,
                "section_type": section_type,
                "source": data_source,
                "chunk_index": chunk_index,
                "rxcui": rxcui,
                "brand_name": brand_name,
            })

    if not all_ids:
        print("STOP: 0 chunks created — check drugs.json structure.")
        return 0

    print(f"\nSaving {len(all_ids)} vectors in batches...")
    batch_size = 5000
    for i in range(0, len(all_ids), batch_size):
        end = min(i + batch_size, len(all_ids))
        print(f"  batch {i} → {end}...")
        collection.add(
            ids=all_ids[i:end],
            embeddings=all_embeddings[i:end],
            documents=all_documents[i:end],
            metadatas=all_metadatas[i:end],
        )

    print(f"Done. Collection '{coll_name}' now has {collection.count()} chunks.")
    return len(all_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest drugs.json into ChromaDB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the collection before ingesting (recommended when re-running full ingest)",
    )
    args = parser.parse_args()
    ingest_json_to_chroma(reset=args.reset)
