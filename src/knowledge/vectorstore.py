"""
ChromaDB storage for FDA / DDInter / user-scan chunks.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings

from .embedder import BGEEmbedder

logger = logging.getLogger(__name__)

COLLECTION_NAME = "medlabel_chunks"


class MedLabelVectorStore:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedder: Optional[BGEEmbedder] = None,
    ) -> None:
        self.persist_dir = persist_dir or os.getenv("CHROMA_DB_PATH", "./medlabel_db")
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or BGEEmbedder()
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def add_chunks(self, chunks: list[dict[str, Any]], *, skip_existing: bool = True) -> int:
        """
        Add chunks: each {text, metadata} where metadata includes drug_name, section_type, etc.
        """
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        metadatas = [_sanitize_metadata(c.get("metadata", {})) for c in chunks]
        ids = [_chunk_id(m, t) for m, t in zip(metadatas, texts)]
        if skip_existing:
            existing = set(self._collection.get(ids=ids).get("ids") or [])
            new_idx = [i for i, cid in enumerate(ids) if cid not in existing]
            if not new_idx:
                return 0
            texts = [texts[i] for i in new_idx]
            metadatas = [metadatas[i] for i in new_idx]
            ids = [ids[i] for i in new_idx]

        embeddings = self.embedder.embed_documents(texts)
        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(ids)

    def vector_search(
        self,
        query: str,
        *,
        n_results: int = 50,
        where: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Dense retrieval only (no reranking)."""
        q_emb = self.embedder.embed_query(query)
        kwargs: dict[str, Any] = {
            "query_embeddings": [q_emb],
            "n_results": min(n_results, max(self.count, 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        if self.count == 0:
            return []
        raw = self._collection.query(**kwargs)
        return _format_results(raw)


def _chunk_id(metadata: dict[str, Any], text: str) -> str:
    payload = text[:500] + "|" + "|".join(
        f"{k}={metadata.get(k, '')}" for k in sorted(metadata)
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma requires scalar metadata values."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    if "drug_name" not in out and out.get("brand_name"):
        out["drug_name"] = out["brand_name"]
    return out


def _format_results(raw: dict) -> list[dict[str, Any]]:
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(
            {
                "text": doc,
                "metadata": meta or {},
                "vector_score": 1.0 - float(dist) if dist is not None else 0.0,
            }
        )
    return out
