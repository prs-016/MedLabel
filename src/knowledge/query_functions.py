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


def _get_embedder(db_path: str = "./drug_db") -> DrugEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = DrugEmbedder(db_path=db_path)
    return _embedder


def _get_reranker() -> TwoStageReranker:
    global _reranker
    if _reranker is None:
        _reranker = TwoStageReranker()
    return _reranker


def _search_and_rerank(
    query:   str,
    *,
    n_fetch: int            = 30,
    top_n:   int            = 5,
    where:   Optional[dict] = None,
    db_path: str            = "./drug_db",
) -> list[RankedChunk]:
    embedder   = _get_embedder(db_path)
    reranker   = _get_reranker()
    raw_chunks = embedder.query(query, n_results=n_fetch, where=where)
    if not raw_chunks:
        logger.info("No chunks from ChromaDB for query: '%s'", query)
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
    """
    JSON stores drug names uppercase e.g. 'WARFARIN SODIUM'.
    Accept any casing from the caller and convert to match.
    """
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
    db_path:      str           = "./drug_db",
) -> list[dict]:
    """
    General semantic search across all ChromaDB chunks.
    Re-ranked by BGE-M3 -> cross-encoder -> hit-count boost.

    Parameters
    ----------
    query        : free-text query  e.g. "warfarin bleeding risk"
    section_type : filter by section  e.g. "warnings_and_cautions"
    drug_name    : restrict to one drug  e.g. "warfarin"
    source       : "fda_label" | "ddinter2_web"
    top_n        : results to return
    n_fetch      : raw results to pull from ChromaDB before reranking

    Returns
    -------
    List of dicts sorted by final_score descending.
    Keys: text, metadata, final_score, bge_score, cross_score,
          hit_count, distance

    Example
    -------
    >>> results = vector_search("warfarin bleeding risk", top_n=5)
    >>> results = vector_search(
    ...     "atorvastatin muscle damage",
    ...     section_type="boxed_warning",
    ...     drug_name="atorvastatin",
    ... )
    """
    where_clauses: list[dict] = []

    if section_type:
        where_clauses.append({"section_type": {"$eq": section_type}})
    if drug_name:
        where_clauses.append({"drug_name": {"$eq": _normalize_drug_name(drug_name)}})
    if source:
        where_clauses.append({"source": {"$eq": source}})

    where  = _build_where(where_clauses)
    ranked = _search_and_rerank(
        query, n_fetch=n_fetch, top_n=top_n, where=where, db_path=db_path
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
    db_path:      str           = "./drug_db",
) -> list[dict]:
    """
    Drug-drug interaction lookup with reranking.

    Parameters
    ----------
    drug_a       : primary drug  e.g. "warfarin"
    drug_b       : optional second drug  e.g. "aspirin"
                   None -> all interactions for drug_a
    min_severity : 0=any  1=Minor  2=Moderate  3=Major
    top_n        : results to return
    n_fetch      : raw ChromaDB fetch count before reranking

    Returns
    -------
    List of interaction dicts sorted by final_score.

    Example
    -------
    >>> results = interaction_check("warfarin", "aspirin", min_severity=2)
    >>> results = interaction_check("warfarin", min_severity=3)
    """
    if drug_b:
        query = f"{drug_a} {drug_b} drug interaction side effects"
    else:
        query = f"{drug_a} drug interactions contraindications"

    where_clauses: list[dict] = [
        {"section_type": {"$in": ["drug_drug_interaction", "drug_interactions"]}}
    ]

    if min_severity > 0:
        where_clauses.append({"severity_score": {"$gte": min_severity}})

    if drug_b:
        where_clauses.append({
            "drug_name": {
                "$in": [
                    _normalize_drug_name(drug_a),
                    _normalize_drug_name(drug_b),
                ]
            }
        })
    else:
        where_clauses.append(
            {"drug_name": {"$eq": _normalize_drug_name(drug_a)}}
        )

    where  = _build_where(where_clauses)
    ranked = _search_and_rerank(
        query, n_fetch=n_fetch, top_n=top_n, where=where, db_path=db_path
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
    db_path:       str           = "./drug_db",
) -> list[dict]:
    """
    Adverse event / reaction lookup for a drug.

    Searches: adverse_reactions, warnings_and_cautions,
              boxed_warning, contraindications, overdosage.

    Parameters
    ----------
    drug_name    : drug to look up  e.g. "warfarin"
    reaction_hint: specific reaction to focus on
                   e.g. "bleeding"  "liver toxicity"
    top_n        : results to return
    n_fetch      : raw ChromaDB fetch count before reranking

    Returns
    -------
    List of dicts sorted by final_score.

    Example
    -------
    >>> results = adverse_events("warfarin", reaction_hint="bleeding")
    >>> results = adverse_events("metformin", reaction_hint="lactic acidosis")
    """
    if reaction_hint:
        query = f"{drug_name} adverse reaction {reaction_hint} side effects"
    else:
        query = f"{drug_name} adverse reactions side effects warnings"

    where_clauses: list[dict] = [
        {
            "section_type": {
                "$in": [
                    "adverse_reactions",
                    "warnings_and_cautions",
                    "boxed_warning",
                    "contraindications",
                    "overdosage",
                ]
            }
        },
        {"drug_name": {"$eq": _normalize_drug_name(drug_name)}},
    ]

    where  = _build_where(where_clauses)
    ranked = _search_and_rerank(
        query, n_fetch=n_fetch, top_n=top_n, where=where, db_path=db_path
    )
    return _format_results(ranked)