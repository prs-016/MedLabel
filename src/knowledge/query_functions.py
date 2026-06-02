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
import os
from typing import Optional

from knowledge.chroma_config import get_chroma_db_path
from knowledge.embedder import DrugEmbedder
from knowledge.reranker import TwoStageReranker, RankedChunk

logger = logging.getLogger(__name__)

_DEFAULT_DB = get_chroma_db_path()

# Section types used for interaction_check (primary + fallback).
_INTERACTION_SECTIONS = [
    "drug_drug_interaction",
    "drug_interactions",
    "interaction_summary",
    "drug_disease_interaction",
    "warnings_and_cautions",
    "warnings",
    "contraindications",
    "precautions",
    "boxed_warning",
]


def _resolve_db_path(db_path: Optional[str] = None) -> str:
    """Map None or the legacy typo '_DEFAULT_DB' to the real Chroma path."""
    if not db_path or db_path == "_DEFAULT_DB":
        return _DEFAULT_DB
    return db_path

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Fine-tuned cross-encoders dedicated to specific query functions.
_VECTOR_SEARCH_MODEL_PATH  = os.path.join(_ROOT_DIR, "models", "vector_search_reranker")
_ADVERSE_EVENTS_MODEL_PATH = os.path.join(_ROOT_DIR, "models", "adverse_events_reranker")
_INTERACTION_MODEL_PATH    = os.path.join(_ROOT_DIR, "models", "medlabel_reranker")

_embedder:               Optional[DrugEmbedder]     = None
_reranker:               Optional[TwoStageReranker] = None
_vector_search_reranker: Optional[TwoStageReranker] = None
_adverse_events_reranker: Optional[TwoStageReranker] = None
_interaction_reranker:   Optional[TwoStageReranker] = None


def _make_finetuned_reranker(
    model_path: str, label: str
) -> TwoStageReranker:
    """Build a TwoStageReranker on a fine-tuned cross-encoder if present,
    else fall back to the base reranker so the function still works.

    The cross-encoder here is fine-tuned for this specific function, so we
    let it dominate the final score (0.8) and keep the generic BGE stage as a
    light prior (0.2) instead of the default 0.6/0.4 split used by the base
    reranker. This stops the untuned BGE (or its offline distance proxy) from
    diluting the task-specific signal.
    """
    if os.path.isdir(model_path):
        logger.info("%s: using fine-tuned reranker at %s", label, model_path)
        return TwoStageReranker(
            cross_encoder_model=model_path,
            bge_weight=0.2,
            cross_weight=0.8,
        )
    logger.warning(
        "%s: fine-tuned reranker not found at %s -- falling back to base "
        "cross-encoder.", label, model_path
    )
    return TwoStageReranker()


def _get_embedder(db_path: Optional[str] = None) -> DrugEmbedder:
    global _embedder
    path = _resolve_db_path(db_path)
    if _embedder is None:
        _embedder = DrugEmbedder(db_path=path)
    return _embedder


def _get_reranker() -> TwoStageReranker:
    global _reranker
    if _reranker is None:
        _reranker = TwoStageReranker()
    return _reranker


def _get_vector_search_reranker() -> TwoStageReranker:
    """Reranker for vector_search (fine-tuned model if available)."""
    global _vector_search_reranker
    if _vector_search_reranker is None:
        _vector_search_reranker = _make_finetuned_reranker(
            _VECTOR_SEARCH_MODEL_PATH, "vector_search"
        )
    return _vector_search_reranker


def _get_adverse_events_reranker() -> TwoStageReranker:
    """Reranker for adverse_events (fine-tuned model if available)."""
    global _adverse_events_reranker
    if _adverse_events_reranker is None:
        _adverse_events_reranker = _make_finetuned_reranker(
            _ADVERSE_EVENTS_MODEL_PATH, "adverse_events"
        )
    return _adverse_events_reranker


def _get_interaction_reranker() -> TwoStageReranker:
    """Reranker for interaction_check (Akshay medlabel_reranker if available)."""
    global _interaction_reranker
    if _interaction_reranker is None:
        _interaction_reranker = _make_finetuned_reranker(
            _INTERACTION_MODEL_PATH, "interaction_check"
        )
    return _interaction_reranker


def _search_and_rerank(
    query:    str,
    *,
    n_fetch:  int                        = 30,
    top_n:    int                        = 5,
    where:    Optional[dict]             = None,
    db_path:  Optional[str]              = None,
    reranker: Optional[TwoStageReranker] = None,
) -> list[RankedChunk]:
    embedder   = _get_embedder(db_path)
    reranker   = reranker or _get_reranker()
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


_drug_name_index: Optional[list] = None


def _all_stored_drug_names(db_path: Optional[str] = None) -> list:
    """Distinct drug_name / brand_name values in the collection (cached)."""
    global _drug_name_index
    if _drug_name_index is None:
        emb  = _get_embedder(db_path)
        data = emb.collection.get(include=["metadatas"])
        names: set[str] = set()
        for m in data.get("metadatas", []) or []:
            if not m:
                continue
            for key in ("drug_name", "brand_name"):
                val = (m.get(key) or "").strip()
                if val:
                    names.add(val)
        _drug_name_index = sorted(names)
    return _drug_name_index


def clear_drug_name_index_cache() -> None:
    """Call after re-ingest so filters see new brand labels."""
    global _drug_name_index
    _drug_name_index = None


def _matching_drug_names(drug_name: str, db_path: Optional[str] = None) -> list:
    """
    Full stored names (e.g. 'WARFARIN SODIUM') whose text contains the
    given short/generic name (e.g. 'warfarin'). Stored names are full
    label titles, so an exact match on the generic name finds nothing.
    """
    token = _normalize_drug_name(drug_name)
    return [n for n in _all_stored_drug_names(db_path) if token in n.upper()]


def _resolve_drug_filter(
    drug_name: str, db_path: Optional[str] = None
) -> Optional[dict]:
    """
    Build a ChromaDB drug_name $in filter for one drug query.
    Returns None when nothing matches, so the search falls back to a
    pure semantic lookup rather than returning zero results.

    OTC brands (DayQuil, NyQuil, …) use brand-only chunks when present;
    ingredient labels are used only if no brand label exists in Chroma.
    """
    from knowledge.otc_brands import canonical_brand_for_query, ingredient_tokens_for_brand

    canonical = canonical_brand_for_query(drug_name)
    if canonical:
        # Exact drug_name row (e.g. "NyQuil") — avoid diluting with all acetaminophen labels.
        exact = [n for n in _matching_drug_names(canonical, db_path) if n == canonical]
        if exact:
            logger.info(
                "Drug filter: brand-only '%s' → %s", drug_name, exact
            )
            return {"drug_name": {"$in": exact}}

    all_matches: set[str] = set()
    for token in ingredient_tokens_for_brand(drug_name):
        all_matches.update(_matching_drug_names(token, db_path))

    if all_matches:
        return {"drug_name": {"$in": sorted(all_matches)}}

    logger.warning(
        "No stored drug_name contains '%s' (or OTC ingredients) "
        "-- searching without drug filter.",
        _normalize_drug_name(drug_name),
    )
    return None


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


def _is_low_quality_ddi(text: str, metadata: dict) -> bool:
    """Drop empty DDInter rows (Unknown severity, no real interaction text)."""
    if metadata.get("section_type") != "drug_drug_interaction":
        return False
    if metadata.get("source") != "ddinter2_web":
        return False
    t = (text or "").strip().lower()
    if not t:
        return True
    # e.g. "Drug interaction [Unknown]: Acetaminophen + Ibuprofen. - Management: -"
    if metadata.get("severity_label", "").lower() == "unknown":
        boilerplate = (
            t.replace("drug interaction [unknown]:", "")
            .replace("- management: -", "")
            .replace("management: -", "")
        )
        if len(boilerplate.strip()) < 50:
            return True
    if t.endswith("management: -") or t.endswith("management:- "):
        body = t.split("management:")[0]
        if len(body.strip()) < 80:
            return True
    return False


def _drop_low_quality_ddi(chunks: list[RankedChunk]) -> list[RankedChunk]:
    kept = [
        c for c in chunks
        if not _is_low_quality_ddi(c.text, c.metadata or {})
    ]
    if len(kept) < len(chunks):
        logger.info(
            "interaction_check: dropped %d low-quality DDInter chunk(s).",
            len(chunks) - len(kept),
        )
    return kept


def _dedupe_ranked(chunks: list[RankedChunk]) -> list[RankedChunk]:
    seen: set[str] = set()
    out: list[RankedChunk] = []
    for c in chunks:
        key = (c.text or "")[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _supplement_interaction_pair(
    drug_a: str,
    drug_b: str,
    query: str,
    *,
    path: str,
    top_n: int,
    n_fetch: int,
    reranker: TwoStageReranker,
) -> list[RankedChunk]:
    """
    When pair-specific DDI text is missing, pull warnings/interactions for
    each drug separately so Grok can answer from label safety sections.
    """
    sections = [
        "drug_interactions", "warnings_and_cautions", "overdosage",
        "contraindications", "boxed_warning", "drug_disease_interaction",
    ]
    combo_q = (
        f"{drug_a} {drug_b} taking together combining medicines "
        f"overdose maximum daily dose warnings {query}"
    )
    out: list[RankedChunk] = []
    for drug in (drug_a, drug_b):
        drug_filter = _resolve_drug_filter(drug, path)
        if not drug_filter:
            continue
        where = _build_where([
            drug_filter,
            {"section_type": {"$in": sections}},
        ])
        out.extend(_search_and_rerank(
            combo_q,
            n_fetch=max(10, n_fetch // 2),
            top_n=max(2, top_n // 2),
            where=where,
            db_path=path,
            reranker=reranker,
        ))
    return _dedupe_ranked(out)[:top_n]


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
    db_path:      Optional[str] = None,
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
        drug_filter = _resolve_drug_filter(drug_name, db_path)
        if drug_filter:
            where_clauses.append(drug_filter)
    if source:
        where_clauses.append({"source": {"$eq": source}})

    where  = _build_where(where_clauses)
    ranked = _search_and_rerank(
        query,
        n_fetch=n_fetch,
        top_n=top_n,
        where=where,
        db_path=db_path,
        reranker=_get_vector_search_reranker(),
    )
    return _format_results(ranked)


def _text_scan_for_substance(
    drug_a: str,
    substance: str,
    path: str,
    *,
    top_n: int = 5,
) -> list["RankedChunk"]:
    """
    Fetch all stored chunks for drug_a and return those that textually
    mention substance (e.g. 'alcohol').  Used when substance is not a
    drug in the DB so normal drug-filter retrieval misses it.
    """
    emb = _get_embedder(path)
    drug_filter = _resolve_drug_filter(drug_a, path)
    if not drug_filter:
        return []

    substance_lower = substance.lower()
    # Pull up to 100 chunks for drug_a across all sections
    raw = emb.query(
        f"{drug_a} {substance} warnings interactions risk precautions",
        n_results=100,
        where=drug_filter,
    )
    hits = [c for c in raw if substance_lower in c["text"].lower()]
    if not hits:
        return []

    # Convert to RankedChunk and score via cross-encoder
    from knowledge.reranker import RankedChunk
    query_str = f"{drug_a} {substance} interaction warnings"
    chunks_as_rc = [
        RankedChunk(
            text=c["text"],
            metadata=c["metadata"],
            distance=c["distance"],
            bge_score=0.0,
            cross_score=0.0,
            hit_count=0,
            final_score=0.0,
        )
        for c in hits
    ]
    reranker = _get_interaction_reranker()
    ranked = reranker.rerank(query=query_str, chunks=[
        {"text": rc.text, "metadata": rc.metadata, "distance": rc.distance}
        for rc in chunks_as_rc
    ], top_n=top_n)
    return ranked


# ---------------------------------------------------------------------------
# 2. interaction_check
# ---------------------------------------------------------------------------

def interaction_check(
    drug_a:       str,
    drug_b:       Optional[str] = None,
    *,
    question:     Optional[str] = None,
    min_severity: int           = 0,
    top_n:        int           = 5,
    n_fetch:      int           = 30,
    db_path:      Optional[str] = None,
) -> list[dict]:
    """
    Drug-drug interaction lookup with reranking.

    Parameters
    ----------
    drug_a       : primary drug  e.g. "warfarin"
    drug_b       : optional second drug  e.g. "aspirin"
                   None -> all interactions for drug_a
    question     : the user's raw question  e.g. "can I take this with alcohol?"
                   Folded into the rerank query so the cross-encoder can
                   prioritise the chunk that actually answers it (e.g. the
                   alcohol/liver warning) instead of a generic warnings section.
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
    # Ensure drug_a is the one with DB entries; swap if drug_b is the known drug.
    if drug_b:
        a_names = _matching_drug_names(drug_a)
        b_names = _matching_drug_names(drug_b)
        if not a_names and b_names:
            drug_a, drug_b = drug_b, drug_a

    hint = (question or "").strip()
    if hint:
        # The user's own words drive relevance; keep only the drug name(s) as
        # context so generic "interactions/warnings" boilerplate doesn't pull
        # off-topic sections to the top.
        drugs = f"{drug_a} {drug_b}" if drug_b else drug_a
        query = f"{hint} {drugs}"
    elif drug_b:
        query = (
            f"{drug_a} {drug_b} drug interaction contraindications "
            "warnings combining medicines"
        )
    else:
        query = f"{drug_a} drug interactions contraindications warnings"

    path = _resolve_db_path(db_path)
    ic_reranker = _get_interaction_reranker()

    if drug_b:
        names = list({
            *_matching_drug_names(drug_a, path),
            *_matching_drug_names(drug_b, path),
        })
    else:
        names = _matching_drug_names(drug_a, path)

    drug_filter: Optional[dict] = None
    if names:
        drug_filter = {"drug_name": {"$in": names}}
    elif drug_a:
        drug_filter = _resolve_drug_filter(drug_a, path)
        if not drug_filter:
            logger.warning(
                "interaction_check: no stored drug_name matched '%s'/'%s' -- "
                "searching without drug filter.", drug_a, drug_b
            )

    severity_clause: Optional[dict] = None
    if min_severity > 0:
        severity_clause = {"severity_score": {"$gte": min_severity}}

    def _run(section_types: Optional[list[str]]) -> list[RankedChunk]:
        clauses: list[dict] = []
        if section_types:
            clauses.append({"section_type": {"$in": section_types}})
        if severity_clause:
            clauses.append(severity_clause)
        if drug_filter:
            clauses.append(drug_filter)
        where = _build_where(clauses)
        return _search_and_rerank(
            query, n_fetch=n_fetch, top_n=top_n, where=where, db_path=path,
            reranker=ic_reranker,
        )

    ranked = _run(_INTERACTION_SECTIONS)
    if not ranked and drug_filter:
        logger.info(
            "interaction_check: no interaction-section hits for '%s'/'%s' "
            "-- retrying with drug filter only.", drug_a, drug_b
        )
        clauses = [drug_filter]
        if severity_clause:
            clauses.append(severity_clause)
        ranked = _search_and_rerank(
            query,
            n_fetch=n_fetch,
            top_n=top_n,
            where=_build_where(clauses),
            db_path=path,
            reranker=ic_reranker,
        )
    if not ranked:
        logger.info(
            "interaction_check: broad semantic fallback for '%s'/'%s'.",
            drug_a, drug_b,
        )
        ranked = _search_and_rerank(
            query, n_fetch=n_fetch, top_n=top_n, where=None, db_path=path,
            reranker=ic_reranker,
        )

    if drug_b:
        pair_hits = _drop_low_quality_ddi(
            _filter_by_drug_pair(ranked, drug_a, drug_b)
        )
        if pair_hits:
            ranked = pair_hits
        else:
            ranked = _drop_low_quality_ddi(ranked)
    else:
        ranked = _drop_low_quality_ddi(ranked)

    if drug_b and len(ranked) < top_n:
        extra = _supplement_interaction_pair(
            drug_a, drug_b, query, path=path, top_n=top_n, n_fetch=n_fetch,
            reranker=ic_reranker,
        )
        ranked = _dedupe_ranked(ranked + extra)

    # If drug_b has no DB entry (e.g. "alcohol", "grapefruit"), do a text-scan
    # across all drug_a chunks to surface mentions of the substance.
    if drug_b and not _matching_drug_names(drug_b, path):
        text_hits = _text_scan_for_substance(drug_a, drug_b, path, top_n=top_n)
        if text_hits:
            ranked = _dedupe_ranked(text_hits + ranked)[:top_n]

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
    db_path:       Optional[str] = None,
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
    ]
    drug_filter = _resolve_drug_filter(drug_name, db_path)
    if drug_filter:
        where_clauses.append(drug_filter)

    where  = _build_where(where_clauses)
    ranked = _search_and_rerank(
        query,
        n_fetch=n_fetch,
        top_n=top_n,
        where=where,
        db_path=db_path,
        reranker=_get_adverse_events_reranker(),
    )
    return _format_results(ranked)