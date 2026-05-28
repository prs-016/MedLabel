"""
Adverse event lookup: openFDA FAERS reports + FDA label ``adverse_reactions`` section.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from api.openfda import OpenFDAClient
from api.rxnorm import RxNormClient

logger = logging.getLogger(__name__)


@dataclass
class ResolvedDrug:
    display_name: str
    generic_name: str
    brand_name: str = ""
    rxcui: str = ""


@dataclass
class AdverseEventsResult:
    drug_query: str
    resolved_generic: str
    faers_reports: list[dict[str, Any]] = field(default_factory=list)
    label_snippets: list[str] = field(default_factory=list)
    search_terms_tried: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "drug_query": self.drug_query,
            "resolved_generic": self.resolved_generic,
            "faers_reports": self.faers_reports,
            "label_adverse_reactions": self.label_snippets,
            "search_terms_tried": self.search_terms_tried,
            "warnings": self.warnings,
            "has_data": bool(self.faers_reports or self.label_snippets),
        }

    def format_for_agent(self) -> str:
        lines = [
            f"Adverse events for: {self.resolved_generic or self.drug_query}",
            "",
        ]

        if self.faers_reports:
            lines.append("openFDA FAERS (top reported reactions):")
            for row in self.faers_reports[:10]:
                lines.append(
                    f"  - {row.get('reaction', '?')}: {row.get('count', 0)} reports"
                )
            lines.append("")
        else:
            lines.append("openFDA FAERS: no summary found for this drug name.")
            lines.append("")

        if self.label_snippets:
            lines.append("FDA label (adverse_reactions section):")
            for snip in self.label_snippets[:3]:
                lines.append(f"  - {snip[:500]}")
            lines.append("")
        else:
            lines.append("FDA label: no adverse_reactions section found.")
            lines.append("")

        for w in self.warnings:
            lines.append(f"Warning: {w}")

        if not self.faers_reports and not self.label_snippets:
            lines.append(
                "No adverse event data found. Try the exact generic or brand name from the label."
            )

        lines.append(
            "\n⚠️ FAERS reports are voluntary and do not prove the drug caused the reaction. "
            "This is informational only — consult a healthcare professional."
        )
        return "\n".join(lines)


def resolve_drug_name(
    drug_name: str,
    rxnorm: Optional[RxNormClient] = None,
) -> ResolvedDrug:
    raw = (drug_name or "").strip()
    if not raw:
        return ResolvedDrug(display_name="", generic_name="")

    rx = rxnorm or RxNormClient()
    resolved = rx.resolve_ocr_string(raw)
    generic = resolved.generic_name or raw
    brand = getattr(resolved, "brand_name", "") or ""
    return ResolvedDrug(
        display_name=raw,
        generic_name=generic,
        brand_name=brand,
        rxcui=resolved.rxcui,
    )


def _search_terms(drug: ResolvedDrug) -> list[str]:
    terms: list[str] = []
    for name in (drug.generic_name, drug.display_name, drug.brand_name):
        n = (name or "").strip()
        if n and n.lower() not in {t.lower() for t in terms}:
            terms.append(n)
    return terms


def _fetch_faers(
    openfda: OpenFDAClient,
    terms: list[str],
    limit: int = 10,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Try multiple drug name variants against FAERS."""
    tried: list[str] = []
    for term in terms:
        tried.append(term)
        hits = openfda.get_adverse_events(term, limit=limit)
        if hits:
            return hits, tried
    return [], tried


def _fetch_label_adverse_reactions(
    openfda: OpenFDAClient,
    drug: ResolvedDrug,
) -> list[str]:
    snippets: list[str] = []
    for name in _search_terms(drug):
        label = openfda.get_label(name)
        if not label:
            continue
        for para in label.adverse_reactions:
            text = para.strip()
            if text and text not in snippets:
                snippets.append(text)
        if snippets:
            break
    return snippets


def lookup_adverse_events(
    drug_name: str,
    *,
    scanned_drug: Optional[str] = None,
    openfda: Optional[OpenFDAClient] = None,
    rxnorm: Optional[RxNormClient] = None,
    faers_limit: int = 10,
) -> AdverseEventsResult:
    """
    Look up adverse events for a drug.

    Parameters
    ----------
    drug_name:
        Drug from the user question, or empty to use ``scanned_drug``.
    scanned_drug:
        Drug identified from the scanned label (preferred when user asks about "this" medicine).
    """
    query = (drug_name or "").strip() or (scanned_drug or "").strip()
    if not query:
        raise ValueError("adverse_events requires a drug name or scanned drug context.")

    # "side effects of this" → use scanned drug
    if scanned_drug and _refers_to_scanned_drug(query):
        query = scanned_drug

    resolved = resolve_drug_name(query, rxnorm)
    fda = openfda or OpenFDAClient()
    terms = _search_terms(resolved)

    faers, tried = _fetch_faers(fda, terms, limit=faers_limit)
    label_snips = _fetch_label_adverse_reactions(fda, resolved)

    warnings: list[str] = []
    if not faers and not label_snips:
        warnings.append(f"Searched FAERS/label as: {', '.join(terms) or query}")

    return AdverseEventsResult(
        drug_query=query,
        resolved_generic=resolved.generic_name or query,
        faers_reports=faers,
        label_snippets=label_snips,
        search_terms_tried=tried,
        warnings=warnings,
    )


def _refers_to_scanned_drug(query: str) -> bool:
    q = query.lower()
    return bool(
        re.search(r"\b(this|it|the medicine|the drug|my medicine|scanned)\b", q)
        or "side effect" in q
        or "adverse" in q
    )


def adverse_events_json(
    drug_name: str,
    scanned_drug: Optional[str] = None,
    **kwargs: Any,
) -> str:
    return json.dumps(
        lookup_adverse_events(drug_name, scanned_drug=scanned_drug, **kwargs).to_dict(),
        indent=2,
    )
