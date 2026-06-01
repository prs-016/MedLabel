"""
src/knowledge/reranker.py
==========================
Two-stage reranking pipeline:
  Stage 1 -- BGE-M3 reranker  (FlagEmbedding, optional)
  Stage 2 -- Cross-encoder    (sentence-transformers CrossEncoder)

If FlagEmbedding is unavailable or broken, falls back to
cross-encoder only -- rankings are still correct.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
_LOCAL_BGE_RERANKER = _ROOT / "models" / "cache" / "bge-reranker-v2-m3"

_bge_reranker  = None
# Cache cross-encoders per model name so multiple fine-tuned models
# (e.g. interaction vs vector_search rerankers) can coexist in one process.
_cross_encoders: dict = {}


def _resolve_bge_model_path(model_name: str) -> str:
    """Prefer local checkout so Hugging Face offline mode still works."""
    env_path = os.getenv("MEDLABEL_BGE_RERANKER_PATH", "").strip()
    if env_path and os.path.isdir(env_path):
        return env_path
    if _LOCAL_BGE_RERANKER.is_dir() and any(_LOCAL_BGE_RERANKER.glob("*.safetensors")):
        return str(_LOCAL_BGE_RERANKER)
    return model_name


def _bge_use_fp16() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _get_bge_reranker(model_name: str = _DEFAULT_BGE_RERANKER):
    global _bge_reranker
    if os.getenv("MEDLABEL_SKIP_BGE_RERANKER", "").strip().lower() in (
        "1", "true", "yes",
    ):
        return None
    if _bge_reranker is None:
        try:
            from FlagEmbedding import FlagReranker

            path = _resolve_bge_model_path(model_name)
            logger.info("Loading BGE reranker from %s", path)
            _bge_reranker = FlagReranker(path, use_fp16=_bge_use_fp16())
            logger.info("BGE reranker loaded.")
        except Exception as exc:
            logger.warning(
                "BGE reranker unavailable (%s) -- using cross-encoder only.", exc
            )
            _bge_reranker = None
    return _bge_reranker


def _get_cross_encoder(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
):
    if model_name not in _cross_encoders:
        from sentence_transformers import CrossEncoder
        logger.info("Loading CrossEncoder: %s", model_name)
        _cross_encoders[model_name] = CrossEncoder(model_name)
        logger.info("CrossEncoder loaded: %s", model_name)
    return _cross_encoders[model_name]


@dataclass
class RankedChunk:
    text:        str
    metadata:    dict  = field(default_factory=dict)
    distance:    float = 1.0
    bge_score:   float = 0.0
    cross_score: float = 0.0
    final_score: float = 0.0
    hit_count:   int   = 1


class TwoStageReranker:
    """
    Pipeline
    --------
    vector results (ChromaDB cosine)
        -> Stage 1: BGE-M3 reranker   (top bge_top_k)  [optional]
        -> Stage 2: Cross-encoder     (top cross_top_k)
        -> hit-count boost
        -> final sorted list
    """

    def __init__(
        self,
        bge_model:           str   = "BAAI/bge-reranker-v2-m3",
        cross_encoder_model: str   = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bge_top_k:           int   = 20,
        cross_top_k:         int   = 10,
        bge_weight:          float = 0.4,
        cross_weight:        float = 0.6,
    ):
        self.bge_model           = bge_model
        self.cross_encoder_model = cross_encoder_model
        self.bge_top_k           = bge_top_k
        self.cross_top_k         = cross_top_k
        self.bge_weight          = bge_weight
        self.cross_weight        = cross_weight

    def rerank(
        self,
        query:  str,
        chunks: list[dict],
        top_n:  int = 5,
    ) -> list[RankedChunk]:
        if not chunks:
            return []

        ranked = [
            RankedChunk(
                text     = c["text"],
                metadata = c.get("metadata", {}),
                distance = c.get("distance", 1.0),
            )
            for c in chunks
        ]

        ranked = self._bge_stage(query, ranked)
        ranked = self._cross_encoder_stage(query, ranked)
        ranked = self._combine_scores(ranked)
        ranked = self._apply_hit_count_boost(ranked)
        ranked.sort(key=lambda c: c.final_score, reverse=True)
        return ranked[:top_n]

    def _bge_stage(
        self, query: str, chunks: list[RankedChunk]
    ) -> list[RankedChunk]:
        reranker = _get_bge_reranker(self.bge_model)
        pool     = chunks[: self.bge_top_k]

        if reranker is None:
            # fallback: use inverse distance as proxy bge score
            for c in pool:
                c.bge_score = 1.0 - c.distance
            return pool

        pairs = [[query, c.text] for c in pool]
        try:
            scores = reranker.compute_score(pairs, normalize=True)
            if isinstance(scores, float):
                scores = [scores]
            for chunk, score in zip(pool, scores):
                chunk.bge_score = float(score)
        except Exception as exc:
            logger.error("BGE reranker scoring failed: %s", exc)
            for c in pool:
                c.bge_score = 1.0 - c.distance

        pool.sort(key=lambda c: c.bge_score, reverse=True)
        return pool

    def _cross_encoder_stage(
        self, query: str, chunks: list[RankedChunk]
    ) -> list[RankedChunk]:
        ce    = _get_cross_encoder(self.cross_encoder_model)
        pool  = chunks[: self.cross_top_k]
        pairs = [[query, c.text] for c in pool]
        try:
            scores = ce.predict(pairs)
            for chunk, score in zip(pool, scores):
                chunk.cross_score = float(score)
        except Exception as exc:
            logger.error("CrossEncoder failed: %s", exc)
            for c in pool:
                c.cross_score = c.bge_score
        return pool

    @staticmethod
    def _normalise(values: list[float]) -> list[float]:
        lo, hi = min(values), max(values)
        if hi == lo:
            return [1.0] * len(values)
        return [(v - lo) / (hi - lo) for v in values]

    def _combine_scores(
        self, chunks: list[RankedChunk]
    ) -> list[RankedChunk]:
        if not chunks:
            return chunks
        bge_n   = self._normalise([c.bge_score   for c in chunks])
        cross_n = self._normalise([c.cross_score  for c in chunks])
        for i, chunk in enumerate(chunks):
            chunk.final_score = (
                self.bge_weight   * bge_n[i] +
                self.cross_weight * cross_n[i]
            )
        return chunks

    @staticmethod
    def _apply_hit_count_boost(
        chunks: list[RankedChunk],
    ) -> list[RankedChunk]:
        from collections import Counter
        drug_counts = Counter(
            c.metadata.get("drug_name", "__unknown__")
            for c in chunks
        )
        for chunk in chunks:
            drug            = chunk.metadata.get("drug_name", "__unknown__")
            count           = drug_counts[drug]
            chunk.hit_count = count
            boost           = 1.0 + min(0.1 * (count - 1), 0.5)
            chunk.final_score *= boost
        return chunks