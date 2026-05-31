"""
Fetch FDA labels + DDInter pairs for common OTC medicines and merge into
data/drugs.json.

Usage (from repo root):
    PYTHONPATH=src python src/knowledge/expand_common_drugs.py
    PYTHONPATH=src python src/knowledge/expand_common_drugs.py --skip-ddi
    PYTHONPATH=src python src/knowledge/expand_common_drugs.py --ingest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.ddinter import DDInterClient
from api.openfda import OpenFDAClient
from api.rxnorm import RxNormClient

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "drugs.json")

# Brand names tried when generic openFDA search returns sparse OTC panels.
OTC_BRAND_FALLBACKS: dict[str, list[str]] = {
    "acetaminophen": ["TYLENOL", "TYLENOL EXTRA STRENGTH"],
    "ibuprofen": ["ADVIL", "MOTRIN"],
    "aspirin": ["BAYER ASPIRIN", "BUFFERIN"],
    "diphenhydramine": ["BENADRYL"],
    "loratadine": ["CLARITIN"],
    "cetirizine": ["ZYRTEC"],
    "dextromethorphan": ["ROBITUSSIN"],
    "guaifenesin": ["MUCINEX"],
    "naproxen": ["ALEVE"],
    "omeprazole": ["PRILOSEC"],
    "fluticasone propionate": ["FLONASE"],
}
COMMON_OTC_DRUGS = [
    "acetaminophen",       # Tylenol
    "diphenhydramine",     # Benadryl
    "loratadine",          # Claritin
    "cetirizine",          # Zyrtec
    "dextromethorphan",    # Robitussin DM
    "doxylamine",          # NyQuil (sleep aid)
    "guaifenesin",         # Mucinex
    "pseudoephedrine",     # Sudafed / some cold meds
    "melatonin",
    "bismuth subsalicylate",  # Pepto-Bismol
    "calcium carbonate",      # Tums
    "fluticasone propionate", # Flonase
]

# Refresh thin OTC entries already in the corpus.
REFRESH_FDA = ["ibuprofen", "aspirin"]

# DDInter batches: groups of drugs checked together (max 5 per batch).
DDI_BATCHES = [
    ["acetaminophen", "ibuprofen", "aspirin", "naproxen", "diphenhydramine"],
    ["acetaminophen", "dextromethorphan", "doxylamine", "pseudoephedrine", "guaifenesin"],
    ["loratadine", "cetirizine", "diphenhydramine", "melatonin", "omeprazole"],
    ["ibuprofen", "naproxen", "aspirin", "warfarin", "metformin"],
]


def _normalize_token(name: str) -> str:
    return (name or "").strip().lower()


def _existing_drug_tokens(records: list[dict]) -> set[str]:
    tokens: set[str] = set()
    for item in records:
        meta = item.get("metadata") or {}
        for key in ("drug_name", "brand_name"):
            val = meta.get(key) or ""
            tokens.add(_normalize_token(val))
            for part in val.split():
                tokens.add(_normalize_token(part))
    return tokens


RICH_FDA_SECTIONS = {
    "drug_interactions",
    "warnings_and_cautions",
    "adverse_reactions",
    "contraindications",
    "boxed_warning",
}


def _has_fda_label(records: list[dict], generic: str) -> bool:
    g = _normalize_token(generic)
    for item in records:
        meta = item.get("metadata") or {}
        if meta.get("source") != "fda_label":
            continue
        dn = _normalize_token(meta.get("drug_name", ""))
        if g in dn or dn.startswith(g):
            return True
    return False


def _has_rich_fda_label(records: list[dict], generic: str) -> bool:
    g = _normalize_token(generic)
    sections: set[str] = set()
    for item in records:
        meta = item.get("metadata") or {}
        if meta.get("source") != "fda_label":
            continue
        dn = _normalize_token(meta.get("drug_name", ""))
        if g in dn or dn.startswith(g):
            sections.add(meta.get("section_type", ""))
    return bool(sections & RICH_FDA_SECTIONS)


def _make_fda_record(chunk: dict, index: int) -> dict:
    meta = chunk["metadata"]
    drug = meta.get("drug_name", "UNKNOWN")
    section = meta.get("section_type", "unknown")
    source = meta.get("source", "fda_label")
    return {
        "id": f"{drug}__{section}__{source}__{index}",
        "text": chunk["text"],
        "metadata": meta,
    }


def _chunk_key(item: dict) -> str:
    meta = item.get("metadata") or {}
    payload = (
        f"{meta.get('drug_name')}|{meta.get('section_type')}|"
        f"{meta.get('source')}|{(item.get('text') or '')[:200]}"
    )
    return hashlib.md5(payload.encode()).hexdigest()


def fetch_fda_records(
    drug_names: list[str],
    rxnorm: RxNormClient,
    fda: OpenFDAClient,
    *,
    delay: float = 0.35,
) -> list[dict]:
    out: list[dict] = []
    for name in drug_names:
        resolved = rxnorm.resolve_ocr_string(name)
        query_name = resolved.generic_name if resolved.found else name
        rxcui = resolved.rxcui if resolved.found else ""

        chunks: list[dict] = []
        for candidate in [query_name, name, *OTC_BRAND_FALLBACKS.get(name, [])]:
            chunks = fda.get_label_chunks(candidate)
            if chunks and any(
                c["metadata"].get("section_type") in (
                    "drug_interactions", "warnings_and_cautions",
                    "adverse_reactions", "contraindications",
                )
                for c in chunks
            ):
                break
            if chunks and not chunks:
                chunks = []

        if not chunks:
            for candidate in [query_name, name, *OTC_BRAND_FALLBACKS.get(name, [])]:
                chunks = fda.get_label_chunks(candidate)
                if chunks:
                    break

        if not chunks:
            logger.warning("No FDA label for '%s' (queried as '%s')", name, query_name)
            time.sleep(delay)
            continue

        for i, chunk in enumerate(chunks):
            chunk["metadata"]["rxcui"] = rxcui or chunk["metadata"].get("rxcui", "")
            out.append(_make_fda_record(chunk, i))

        logger.info("FDA: %s → %d sections", query_name, len(chunks))
        time.sleep(delay)
    return out


def fetch_ddi_records(
    batches: list[list[str]],
    ddinter: DDInterClient,
    *,
    delay: float = 1.5,
) -> list[dict]:
    out: list[dict] = []
    seen_ids: set[str] = set()

    for idx, batch in enumerate(batches):
        logger.info("DDInter batch %d/%d: %s", idx + 1, len(batches), batch)
        docs = ddinter.check_interactions_for_session(batch, session_id=f"otc-{idx}")
        added = 0
        for doc in docs:
            doc_id = doc.get("id") or _chunk_key(doc)
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            out.append(doc)
            added += 1
        logger.info("  → %d new DDI records", added)
        time.sleep(delay)
    return out


def merge_records(
    existing: list[dict],
    new_records: list[dict],
    *,
    replace_fda_for: Optional[set[str]] = None,
) -> list[dict]:
    replace_fda_for = replace_fda_for or set()
    replace_tokens = {_normalize_token(d) for d in replace_fda_for}

    if replace_tokens:
        kept = []
        for item in existing:
            meta = item.get("metadata") or {}
            if meta.get("source") != "fda_label":
                kept.append(item)
                continue
            dn = _normalize_token(meta.get("drug_name", ""))
            if any(tok in dn or dn.startswith(tok) for tok in replace_tokens):
                continue
            kept.append(item)
        existing = kept

    seen_ids = {item.get("id") for item in existing if item.get("id")}
    seen_keys = {_chunk_key(item) for item in existing}
    merged = list(existing)

    for item in new_records:
        cid = item.get("id")
        key = _chunk_key(item)
        if cid and cid in seen_ids:
            continue
        if key in seen_keys:
            continue
        merged.append(item)
        if cid:
            seen_ids.add(cid)
        seen_keys.add(key)
    return merged


def expand_common_drugs(
    *,
    skip_fda: bool = False,
    skip_ddi: bool = False,
    refresh_existing: bool = True,
) -> dict:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)

    tokens = _existing_drug_tokens(existing)
    to_fetch: list[str] = []
    for drug in COMMON_OTC_DRUGS:
        if not _has_fda_label(existing, drug) or not _has_rich_fda_label(existing, drug):
            to_fetch.append(drug)
        else:
            logger.info("Skipping FDA fetch for '%s' (rich label present)", drug)

    if refresh_existing:
        for drug in REFRESH_FDA:
            if drug not in to_fetch:
                to_fetch.append(drug)

    rxnorm = RxNormClient()
    fda = OpenFDAClient()
    ddinter = DDInterClient()

    new_fda: list[dict] = []
    new_ddi: list[dict] = []

    if not skip_fda and to_fetch:
        logger.info("Fetching FDA labels for %d drugs...", len(to_fetch))
        new_fda = fetch_fda_records(to_fetch, rxnorm, fda)

    if not skip_ddi:
        logger.info("Fetching DDInter batches...")
        new_ddi = fetch_ddi_records(DDI_BATCHES, ddinter)

    merged = merge_records(
        existing, new_fda + new_ddi, replace_fda_for=set(to_fetch)
    )

    backup = JSON_PATH + ".bak"
    if os.path.isfile(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as src, open(backup, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        logger.info("Backed up existing JSON → %s", backup)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    return {
        "before": len(existing),
        "after": len(merged),
        "fda_added": len(new_fda),
        "ddi_added": len(new_ddi),
        "fda_fetched_for": to_fetch,
    }


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Add common OTC drugs to data/drugs.json")
    parser.add_argument("--skip-fda", action="store_true", help="Skip openFDA label fetch")
    parser.add_argument("--skip-ddi", action="store_true", help="Skip DDInter fetch")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Re-ingest drugs.json into medlabel_db after expanding (--reset)",
    )
    args = parser.parse_args(argv)

    stats = expand_common_drugs(skip_fda=args.skip_fda, skip_ddi=args.skip_ddi)
    print(
        f"\nDone. records {stats['before']} → {stats['after']} "
        f"(+{stats['fda_added']} FDA, +{stats['ddi_added']} DDInter)"
    )
    if stats["fda_fetched_for"]:
        print("FDA fetched for:", ", ".join(stats["fda_fetched_for"]))

    if args.ingest:
        from knowledge.ingest_data import ingest_json_to_chroma
        print("\nRe-ingesting into ChromaDB (this may take ~30 min)...")
        ingest_json_to_chroma(reset=True)


if __name__ == "__main__":
    main()
