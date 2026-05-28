"""
Drug–drug interaction checking: DDInter CSV + FDA label ``drug_interactions`` section.

This module implements the team's ``interaction_check`` tool (Student 3 task).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from api.ddinter import DDInterClient, get_demo_interactions
from api.openfda import OpenFDAClient
from api.rxnorm import RxNormClient
from knowledge.retrieval import HitCount, RetrievalPipeline, RetrievalResult, format_chunks_for_agent

logger = logging.getLogger(__name__)


@dataclass
class ResolvedDrugPair:
    drug_a_display: str
    drug_b_display: str
    drug_a_generic: str
    drug_b_generic: str
    rxcui_a: str = ""
    rxcui_b: str = ""


@dataclass
class InteractionCheckResult:
    drug_a: str
    drug_b: str
    ddinter_hits: list[dict[str, Any]] = field(default_factory=list)
    fda_snippets: list[dict[str, Any]] = field(default_factory=list)
    reranked_chunks: list[dict[str, Any]] = field(default_factory=list)
    hit_counts: list[dict[str, Any]] = field(default_factory=list)
    used_demo_data: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "drug_a": self.drug_a,
            "drug_b": self.drug_b,
            "ddinter": self.ddinter_hits,
            "fda_label_interactions": self.fda_snippets,
            "reranked_chunks": self.reranked_chunks,
            "hit_counts": self.hit_counts,
            "used_demo_data": self.used_demo_data,
            "warnings": self.warnings,
            "has_interaction_data": bool(
                self.ddinter_hits or self.fda_snippets or self.reranked_chunks
            ),
        }

    def format_for_agent(self) -> str:
        """Plain-text summary for LangChain tool output."""
        lines = [
            f"Interaction check: {self.drug_a} + {self.drug_b}",
            "",
        ]
        if self.used_demo_data:
            lines.append(
                "(Note: DDInter CSV not found — showing built-in demo pairs only. "
                "Add CSVs to data/ddinter/ for full coverage.)"
            )
            lines.append("")

        if self.ddinter_hits:
            lines.append("DDInter:")
            for hit in self.ddinter_hits:
                level = hit.get("level") or "unknown severity"
                lines.append(f"  - [{level}] {hit.get('drug_a')} + {hit.get('drug_b')}")
                if hit.get("mechanism"):
                    lines.append(f"    Mechanism: {hit['mechanism'][:300]}")
                if hit.get("management"):
                    lines.append(f"    Management: {hit['management'][:300]}")
            lines.append("")
        else:
            lines.append("DDInter: no matching pair in database.")
            lines.append("")

        if self.reranked_chunks:
            rag_result = RetrievalResult(
                query=f"{self.drug_a} + {self.drug_b}",
                chunks=self.reranked_chunks,
                hit_counts=[
                    HitCount(
                        key=h.get("key", ""),
                        label=h.get("label", ""),
                        count=int(h.get("count", 0)),
                        top_score=float(h.get("top_score", 0)),
                    )
                    for h in self.hit_counts
                ],
            )
            lines.append(
                format_chunks_for_agent(
                    rag_result,
                    title="RAG evidence (BGE-M3 → BGE reranker → cross-encoder)",
                )
            )
            lines.append("")
        elif self.fda_snippets:
            lines.append("FDA label (drug_interactions section):")
            for snip in self.fda_snippets:
                lines.append(f"  - [{snip.get('drug_label', '?')}] {snip.get('text', '')[:400]}")
            lines.append("")
        else:
            lines.append("FDA label / RAG: no interaction passages found for this pair.")
            lines.append("")

        for w in self.warnings:
            lines.append(f"Warning: {w}")

        if not self.ddinter_hits and not self.fda_snippets and not self.used_demo_data:
            lines.append(
                "No interaction data found. Advise user to consult a pharmacist or prescriber."
            )

        lines.append(
            "\n⚠️ Informational only — not medical advice. Verify with label and pharmacist."
        )
        return "\n".join(lines)


def parse_interaction_query(
    query: str,
    scanned_drug: Optional[str] = None,
) -> tuple[str, str]:
    """
    Parse tool input into (drug_a, drug_b).

    Accepted formats:
      - ``"ibuprofen|acetaminophen"``
      - ``"Can I take ibuprofen with acetaminophen?"`` (best-effort)
      - With ``scanned_drug`` set: query is only the *other* drug name
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("interaction_check requires a drug name or pair in the query.")

    if "|" in q:
        parts = [p.strip() for p in q.split("|", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            return parts[0], parts[1]

    # "… with ibuprofen?" / "Can I take this with X"
    m = re.search(
        r"(?:with|and|plus)\s+([a-zA-Z][a-zA-Z0-9\s\-/']{0,60}?)(?:\?|$|\.)",
        q,
        re.IGNORECASE,
    )
    if m:
        other = m.group(1).strip()
        if scanned_drug:
            return scanned_drug.strip(), other
        return q, other  # noqa: unlikely path

    m = re.search(
        r"(?:take|use|combine)?\s*(.+?)\s+(?:with|and|plus)\s+(.+?)(?:\?|$)",
        q,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()

    if scanned_drug:
        # Agent passed only the other drug name (e.g. "ibuprofen")
        if len(q.split()) <= 4 and "?" not in q:
            return scanned_drug.strip(), q
        raise ValueError(
            "Could not parse the other drug name. Try: 'ibuprofen' or 'Can I take this with ibuprofen?'"
        )

    raise ValueError(
        "Could not parse two drugs. Use format: 'drug_a|drug_b' or set scanned drug context."
    )


def resolve_drug_names(
    drug_a: str,
    drug_b: str,
    rxnorm: Optional[RxNormClient] = None,
) -> ResolvedDrugPair:
    rx = rxnorm or RxNormClient()
    ra = rx.resolve_ocr_string(drug_a)
    rb = rx.resolve_ocr_string(drug_b)
    return ResolvedDrugPair(
        drug_a_display=drug_a,
        drug_b_display=drug_b,
        drug_a_generic=ra.generic_name or drug_a,
        drug_b_generic=rb.generic_name or drug_b,
        rxcui_a=ra.rxcui,
        rxcui_b=rb.rxcui,
    )


def _fda_interaction_snippets(
    openfda: OpenFDAClient,
    pair: ResolvedDrugPair,
) -> list[dict[str, Any]]:
    snippets: list[dict[str, Any]] = []
    needles = {
        _norm_token(pair.drug_b_display),
        _norm_token(pair.drug_b_generic),
        _norm_token(pair.drug_a_display),
        _norm_token(pair.drug_a_generic),
    }
    needles.discard("")

    for label_drug, other_needles in (
        (pair.drug_a_generic, {_norm_token(pair.drug_b_generic), _norm_token(pair.drug_b_display)}),
        (pair.drug_b_generic, {_norm_token(pair.drug_a_generic), _norm_token(pair.drug_a_display)}),
    ):
        label = openfda.get_label(label_drug)
        if not label:
            continue
        for para in label.drug_interactions:
            text = para.strip()
            if not text:
                continue
            lower = text.lower()
            if any(n and n in lower for n in other_needles):
                snippets.append(
                    {
                        "drug_label": label_drug,
                        "brand_name": label.brand_name,
                        "generic_name": label.generic_name,
                        "text": text,
                        "source": "fda_label",
                    }
                )
    return snippets


def _norm_token(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _lookup_ddinter(
    ddinter: DDInterClient,
    pair: ResolvedDrugPair,
) -> tuple[list[dict[str, Any]], bool]:
    """Return (records as dicts, used_demo_fallback)."""
    names_a = [pair.drug_a_generic, pair.drug_a_display]
    names_b = [pair.drug_b_generic, pair.drug_b_display]
    hits: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for na in names_a:
        for nb in names_b:
            if not na or not nb:
                continue
            for rec in ddinter.lookup_pair(na, nb):
                key = _pair_key(rec.drug_a, rec.drug_b)
                if key not in seen:
                    seen.add(key)
                    hits.append(rec.to_dict())

    if hits:
        return hits, False

    if ddinter.has_data():
        return [], False

    demo_seen: set[tuple[str, str]] = set()
    demo_dicts: list[dict[str, Any]] = []
    for na in names_a:
        for nb in names_b:
            if not na or not nb:
                continue
            for rec in get_demo_interactions(na, nb):
                key = _pair_key(rec.drug_a, rec.drug_b)
                if key not in demo_seen:
                    demo_seen.add(key)
                    demo_dicts.append(rec.to_dict())
    return demo_dicts, bool(demo_dicts)


def _pair_key(a: str, b: str) -> tuple[str, str]:
    na, nb = a.lower(), b.lower()
    return (na, nb) if na <= nb else (nb, na)


def _retrieve_interaction_rag(
    pair: ResolvedDrugPair,
    pipeline: Optional[RetrievalPipeline] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """
    BGE-M3 → BGE reranker → cross-encoder over ChromaDB chunks.
    Returns (reranked_chunks, hit_count_dicts, warnings).
    """
    query = (
        f"{pair.drug_a_generic} {pair.drug_b_generic} "
        "drug drug interaction contraindication concomitant use"
    )
    from knowledge.vectorstore import MedLabelVectorStore

    store = MedLabelVectorStore()
    if store.count == 0:
        return [], [], ["ChromaDB is empty. Run: python scripts/ingest_to_chromadb.py"]

    pipe = pipeline or RetrievalPipeline(store=store)
    result = pipe.retrieve(query, vector_k=50, bge_top_k=20, cross_top_k=10)
    hit_dicts = [
        {
            "key": h.key,
            "label": h.label,
            "count": h.count,
            "top_score": h.top_score,
        }
        for h in result.hit_counts
    ]
    return result.chunks, hit_dicts, result.warnings


def interaction_check(
    query: str,
    scanned_drug: Optional[str] = None,
    *,
    ddinter: Optional[DDInterClient] = None,
    openfda: Optional[OpenFDAClient] = None,
    rxnorm: Optional[RxNormClient] = None,
    retrieval: Optional[RetrievalPipeline] = None,
) -> InteractionCheckResult:
    """
    Check interactions between two drugs using DDInter + FDA labels.

    Parameters
    ----------
    query:
        Other drug name, or ``drug_a|drug_b`` pair.
    scanned_drug:
        Drug from the user's scanned label (drug_a) when query is only the second drug.
    """
    drug_a_raw, drug_b_raw = parse_interaction_query(query, scanned_drug)
    resolved = resolve_drug_names(drug_a_raw, drug_b_raw, rxnorm)

    dd = ddinter or DDInterClient()
    fda = openfda or OpenFDAClient()

    ddinter_hits, used_demo = _lookup_ddinter(dd, resolved)
    rag_chunks, hit_counts, rag_warnings = _retrieve_interaction_rag(
        resolved, pipeline=retrieval
    )
    fda_hits = _fda_interaction_snippets(fda, resolved) if not rag_chunks else []

    warnings: list[str] = list(rag_warnings)
    if not dd.has_data() and not used_demo:
        warnings.append(
            "DDInter CSV files not loaded — place files in data/ddinter/ (see README)."
        )

    return InteractionCheckResult(
        drug_a=resolved.drug_a_generic,
        drug_b=resolved.drug_b_generic,
        ddinter_hits=ddinter_hits,
        fda_snippets=fda_hits,
        reranked_chunks=rag_chunks,
        hit_counts=hit_counts,
        used_demo_data=used_demo,
        warnings=warnings,
    )


def interaction_check_json(
    query: str,
    scanned_drug: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """JSON string for programmatic use."""
    return json.dumps(
        interaction_check(query, scanned_drug=scanned_drug, **kwargs).to_dict(),
        indent=2,
    )
