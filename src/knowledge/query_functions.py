"""
src/knowledge/query_functions.py
=================================
Three public query functions on top of ChromaDB + TwoStageReranker.
    vector_search(query, ...)              -- semantic search over all chunks
    interaction_check(drug_a, drug_b, ...) -- drug-drug interaction lookup
    adverse_events(drug_name, ...)         -- adverse reaction / event lookup
"""
from __future__ import annotations
import logging
from typing import Optional
from knowledge.embedder import DrugEmbedder
from knowledge.reranker import TwoStageReranker, RankedChunk

logger = logging.getLogger(__name__)

_embedder: Optional[DrugEmbedder]     = None
_reranker: Optional[TwoStageReranker] = None


def _get_embedder(db_path: str = "./medlabel_db") -> DrugEmbedder:
    global _embedder
    if _embedder is None or _embedder.db_path != db_path:
        _embedder = DrugEmbedder(db_path=db_path)
    return _embedder


def _get_reranker() -> TwoStageReranker:
    global _reranker
    if _reranker is None:
        _reranker = TwoStageReranker()
    return _reranker


def _search_and_rerank(
    query:            str,
    *,
    n_fetch:          int            = 30,
    top_n:            int            = 5,
    where:            Optional[dict] = None,
    db_path:          str            = "./medlabel_db",
    drug_name_filter: Optional[str]  = None,
) -> list[RankedChunk]:
    embedder   = _get_embedder(db_path)
    reranker   = _get_reranker()
    raw_chunks = embedder.query(query, n_results=n_fetch, where=where)
    if not raw_chunks:
        logger.info("No chunks from ChromaDB for query: '%s'", query)
        return []
    if drug_name_filter:
        needle     = drug_name_filter.lower()
        raw_chunks = [
            c for c in raw_chunks
            if needle in c.get("metadata", {}).get("drug_name", "").lower()
        ]
    if not raw_chunks:
        return []
    return reranker.rerank(query=query, chunks=raw_chunks, top_n=top_n)


def _format_results(ranked: list[RankedChunk]) -> list[dict]:
    return [
        {
            "text":        r.text,
            "metadata":    r.metadata,
            "final_score": round(r.final_score, 4),
            "bge_score":   round(r.bge_score,   4),
            "cross_score": round(r.cross_score,  4),
            "hit_count":   r.hit_count,
            "distance":    round(r.distance,     4),
        }
        for r in ranked
    ]


def _build_where(clauses: list[dict]) -> Optional[dict]:
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _normalize_drug_name(name: str) -> str:
    return name.upper()


def _filter_by_drug_pair(
    chunks: list[RankedChunk],
    drug_a: str,
    drug_b: str,
) -> list[RankedChunk]:
    a_lower = drug_a.lower()
    b_lower = drug_b.lower()

    def _mentions_both(c: RankedChunk) -> bool:
        combined = (c.text + str(c.metadata)).lower()
        return a_lower in combined and b_lower in combined

    filtered = [c for c in chunks if _mentions_both(c)]
    if not filtered:
        logger.debug(
            "interaction_check: no chunks mention both '%s' and '%s' "
            "-- returning unrestricted results.", drug_a, drug_b
        )
        return chunks
    return filtered


# ---------------------------------------------------------------------------
# 1. vector_search
# ---------------------------------------------------------------------------
def vector_search(
    query:        str,
    *,
    section_type: Optional[str] = None,
    drug_name:    Optional[str] = None,
    source:       Optional[str] = None,
    top_n:        int           = 5,
    n_fetch:      int           = 30,
    db_path:      str           = "./medlabel_db",
) -> list[dict]:
    where_clauses: list[dict] = []
    if section_type:
        where_clauses.append({"section_type": {"$eq": section_type}})
    if source:
        where_clauses.append({"source": {"$eq": source}})
    where  = _build_where(where_clauses)
    ranked = _search_and_rerank(
        query, n_fetch=n_fetch, top_n=top_n, where=where,
        db_path=db_path, drug_name_filter=drug_name
    )
    return _format_results(ranked)


# ---------------------------------------------------------------------------
# 2. interaction_check
# ---------------------------------------------------------------------------
def interaction_check(
    drug_a:       str,
    drug_b:       Optional[str] = None,
    *,
    min_severity: int           = 0,
    top_n:        int           = 5,
    n_fetch:      int           = 30,
    db_path:      str           = "./medlabel_db",
) -> list[dict]:
    if drug_b:
        query = f"{drug_a} {drug_b} drug interaction side effects"
    else:
        query = f"{drug_a} drug interactions contraindications"

    ranked = _search_and_rerank(
        query, n_fetch=n_fetch, top_n=top_n, where=None,
        db_path=db_path, drug_name_filter=drug_a
    )
    if drug_b:
        ranked = _filter_by_drug_pair(ranked, drug_a, drug_b)
    return _format_results(ranked[:top_n])


# ---------------------------------------------------------------------------
# 3. adverse_events
# ---------------------------------------------------------------------------
def adverse_events(
    drug_name:     str,
    *,
    reaction_hint: Optional[str] = None,
    top_n:         int           = 5,
    n_fetch:       int           = 30,
    db_path:       str           = "./medlabel_db",
) -> list[dict]:
    if reaction_hint:
        query = f"{drug_name} adverse reaction {reaction_hint} side effects"
    else:
        query = f"{drug_name} adverse reactions side effects warnings"

    ranked = _search_and_rerank(
        query, n_fetch=n_fetch, top_n=top_n, where=None,
        db_path=db_path, drug_name_filter=drug_name
    )
    return _format_results(ranked)
