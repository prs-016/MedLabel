"""
Stage 2 (BGE reranker) and stage 3 (cross-encoder) reranking for retrieved chunks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_FINETUNED_DIR = str(
    Path(__file__).resolve().parents[2] / "models" / "cross_encoder_medlabel"
)


def get_cross_encoder_model_name() -> str:
    """
    Prefer fine-tuned local model when present.

    Set ``CROSS_ENCODER_MODEL`` in ``.env`` to override (path or HuggingFace id).
    """
    env = os.getenv("CROSS_ENCODER_MODEL", "").strip()
    if env:
        return env
    if Path(DEFAULT_FINETUNED_DIR).is_dir():
        return DEFAULT_FINETUNED_DIR
    return DEFAULT_CROSS_ENCODER


class BGEReranker:
    """BGE-M3 reranker (pairwise query–passage scores)."""

    def __init__(self, model_name: str = DEFAULT_BGE_RERANKER) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as e:
                raise ImportError(
                    "Install FlagEmbedding for BGE reranker: pip install FlagEmbedding"
                ) from e
            logger.info("Loading BGE reranker %s …", self.model_name)
            self._model = FlagReranker(self.model_name, use_fp16=False)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        *,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []
        model = self._load()
        pairs = [[query, c.get("text", "")] for c in chunks]
        scores = model.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        out: list[dict[str, Any]] = []
        for chunk, score in ranked[:top_k]:
            c = dict(chunk)
            c["bge_rerank_score"] = float(score)
            out.append(c)
        return out


class CrossEncoderReranker:
    """Cross-encoder reranker on chunk text (final ranking stage)."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or get_cross_encoder_model_name()
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder %s …", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        *,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []
        model = self._load()
        pairs = [(query, c.get("text", "")) for c in chunks]
        scores = model.predict(pairs)
        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        out: list[dict[str, Any]] = []
        for chunk, score in ranked[:top_k]:
            c = dict(chunk)
            c["cross_encoder_score"] = float(score)
            out.append(c)
        return out


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    bge: Optional[BGEReranker] = None,
    cross: Optional[CrossEncoderReranker] = None,
    bge_top_k: int = 20,
    cross_top_k: int = 10,
) -> list[dict[str, Any]]:
    """BGE reranker → cross-encoder reranker."""
    bge_r = bge or BGEReranker()
    cross_r = cross or CrossEncoderReranker()
    stage1 = bge_r.rerank(query, chunks, top_k=bge_top_k)
    return cross_r.rerank(query, stage1, top_k=cross_top_k)
