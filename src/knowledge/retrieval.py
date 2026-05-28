"""
Retrieval pipeline: BGE-M3 dense search → BGE reranker → cross-encoder → hit counts.

Team spec::
  embed (bge-m3) → vector retrieve → bge-reranker → cross-encoder rerank
  → rank sources by number of supporting chunks (hit count).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from knowledge.reranker import BGEReranker, CrossEncoderReranker, rerank_chunks
from knowledge.vectorstore import MedLabelVectorStore

logger = logging.getLogger(__name__)


@dataclass
class HitCount:
    """Aggregated chunk hits for a metadata grouping key."""

    key: str
    label: str
    count: int
    top_score: float = 0.0


@dataclass
class RetrievalResult:
    query: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    hit_counts: list[HitCount] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_results(self) -> bool:
        return bool(self.chunks)


def aggregate_hit_counts(
    chunks: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...] = ("drug_name", "section_type", "source"),
) -> list[HitCount]:
    """
    Count chunks per metadata group; sort by count descending (team: most hits wins).
    """
    counter: Counter[str] = Counter()
    labels: dict[str, str] = {}
    best_score: dict[str, float] = {}

    for ch in chunks:
        meta = ch.get("metadata") or {}
        parts = [str(meta.get(k, "") or "").strip() for k in group_keys]
        parts = [p for p in parts if p]
        if not parts:
            key = "unknown"
            label = "unknown"
        else:
            key = "|".join(parts)
            label = " / ".join(parts)
        counter[key] += 1
        labels[key] = label
        score = float(ch.get("cross_encoder_score") or ch.get("bge_rerank_score") or 0)
        best_score[key] = max(best_score.get(key, 0.0), score)

    hits = [
        HitCount(key=k, label=labels[k], count=c, top_score=best_score.get(k, 0.0))
        for k, c in counter.most_common()
    ]
    return hits


class RetrievalPipeline:
    def __init__(
        self,
        store: Optional[MedLabelVectorStore] = None,
        bge_reranker: Optional[BGEReranker] = None,
        cross_encoder: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.store = store or MedLabelVectorStore()
        self._bge_reranker = bge_reranker
        self._cross_encoder = cross_encoder

    @property
    def bge_reranker(self) -> BGEReranker:
        if self._bge_reranker is None:
            self._bge_reranker = BGEReranker()
        return self._bge_reranker

    @property
    def cross_encoder(self) -> CrossEncoderReranker:
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoderReranker()
        return self._cross_encoder

    def retrieve(
        self,
        query: str,
        *,
        where: Optional[dict[str, Any]] = None,
        vector_k: int = 50,
        bge_top_k: int = 20,
        cross_top_k: int = 10,
    ) -> RetrievalResult:
        warnings: list[str] = []
        if self.store.count == 0:
            warnings.append(
                "ChromaDB is empty. Run: python scripts/ingest_to_chromadb.py"
            )
            return RetrievalResult(query=query, warnings=warnings)

        candidates = self.store.vector_search(query, n_results=vector_k, where=where)
        if not candidates:
            warnings.append("No vector matches for query/metadata filter.")
            return RetrievalResult(query=query, warnings=warnings)

        reranked = rerank_chunks(
            query,
            candidates,
            bge=self.bge_reranker,
            cross=self.cross_encoder,
            bge_top_k=bge_top_k,
            cross_top_k=cross_top_k,
        )
        hits = aggregate_hit_counts(reranked)
        return RetrievalResult(
            query=query,
            chunks=reranked,
            hit_counts=hits,
            warnings=warnings,
        )


def retrieve(
    query: str,
    *,
    where: Optional[dict[str, Any]] = None,
    pipeline: Optional[RetrievalPipeline] = None,
    **kwargs: Any,
) -> RetrievalResult:
    """One-shot retrieval through the full rerank pipeline."""
    pipe = pipeline or RetrievalPipeline()
    return pipe.retrieve(query, where=where, **kwargs)


def format_chunks_for_agent(
    result: RetrievalResult,
    *,
    max_chunks: int = 5,
    title: str = "Retrieved evidence (reranked)",
) -> str:
    lines = [title, ""]
    if result.warnings:
        for w in result.warnings:
            lines.append(f"Note: {w}")
        lines.append("")

    if result.hit_counts:
        lines.append("Sources ranked by chunk hit count:")
        for h in result.hit_counts[:8]:
            lines.append(f"  - {h.label}: {h.count} chunk(s), top score {h.top_score:.3f}")
        lines.append("")

    if not result.chunks:
        lines.append("No matching chunks after reranking.")
        return "\n".join(lines)

    lines.append("Top reranked passages:")
    for i, ch in enumerate(result.chunks[:max_chunks], 1):
        meta = ch.get("metadata") or {}
        drug = meta.get("drug_name", "?")
        section = meta.get("section_type", meta.get("source", "?"))
        score = ch.get("cross_encoder_score", 0)
        text = (ch.get("text") or "")[:400]
        lines.append(f"  {i}. [{drug} / {section}] (score {score:.3f})")
        lines.append(f"     {text}")
    return "\n".join(lines)
