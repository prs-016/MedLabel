"""
Shared ChromaDB path and collection name (from .env).
Default: medlabel_db / medlabel_chunks — Akshay's JSON-ingested store.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")


def get_chroma_db_path() -> str:
    raw = os.getenv("CHROMA_DB_PATH", "./medlabel_db")
    if os.path.isabs(raw):
        return raw
    return str((_ROOT / raw.strip("./")).resolve())


def get_collection_name() -> str:
    return os.getenv("CHROMA_COLLECTION", "medlabel_chunks")
