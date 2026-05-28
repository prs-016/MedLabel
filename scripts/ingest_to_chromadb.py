#!/usr/bin/env python3
"""
Populate ChromaDB for the BGE-M3 + reranker retrieval pipeline.

  export PYTHONPATH=src
  python scripts/ingest_to_chromadb.py
  python scripts/ingest_to_chromadb.py --drug ibuprofen --drug acetaminophen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _setup() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _setup()
    from knowledge.ingest import STARTER_DRUGS, ingest_ddinter, ingest_fda_drugs, ingest_starter_set
    from knowledge.vectorstore import MedLabelVectorStore

    p = argparse.ArgumentParser()
    p.add_argument("--drug", action="append", help="Extra FDA drug to ingest")
    p.add_argument("--skip-ddinter", action="store_true")
    p.add_argument("--skip-fda", action="store_true")
    args = p.parse_args()

    store = MedLabelVectorStore()
    print(f"ChromaDB path: {store.persist_dir} (count before: {store.count})")

    if args.skip_fda and args.skip_ddinter:
        print("Nothing to do.")
        return 0

    if not args.skip_fda:
        drugs = list(STARTER_DRUGS)
        if args.drug:
            drugs.extend(args.drug)
        n = ingest_fda_drugs(drugs, store=store)
        print(f"FDA chunks added: {n}")

    if not args.skip_ddinter:
        n = ingest_ddinter(store=store)
        print(f"DDInter chunks added: {n}")

    print(f"Total chunks in DB: {store.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
