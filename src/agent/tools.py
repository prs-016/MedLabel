"""
LangChain agent tools for MedLabel.

All three tools use the shared RAG pipeline when ChromaDB is populated:
  BGE-M3 embed → vector search → BGE reranker → cross-encoder → hit counts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from agent.adverse_events import lookup_adverse_events
from agent.interaction_check import interaction_check as run_interaction_check
from knowledge.retrieval import format_chunks_for_agent, retrieve

if TYPE_CHECKING:
    from langchain.agents import Tool


def vector_search(query: str, scanned_drug: Optional[str] = None) -> str:
    """
    Search FDA / DDInter chunks via BGE-M3 + BGE reranker + cross-encoder.

    Filters to ``scanned_drug`` when provided.
    """
    try:
        where = None
        if scanned_drug:
            where = {"drug_name": scanned_drug.strip()}
        result = retrieve(query, where=where)
        header = f"vector_search for: {query}"
        if scanned_drug:
            header += f" (drug filter: {scanned_drug})"
        return format_chunks_for_agent(result, title=header)
    except Exception as e:
        return f"vector_search failed: {e}"


def adverse_events(drug_name: str, scanned_drug: Optional[str] = None) -> str:
    """Adverse events: RAG pipeline + openFDA FAERS."""
    try:
        result = lookup_adverse_events(drug_name, scanned_drug=scanned_drug)
        return result.format_for_agent()
    except ValueError as e:
        return f"adverse_events error: {e}"
    except Exception as e:
        return f"adverse_events failed: {e}"


def interaction_check(query: str, scanned_drug: Optional[str] = None) -> str:
    """Drug interactions: DDInter + RAG pipeline (BGE + cross-encoder)."""
    try:
        result = run_interaction_check(query, scanned_drug=scanned_drug)
        return result.format_for_agent()
    except ValueError as e:
        return f"interaction_check error: {e}"
    except Exception as e:
        return f"interaction_check failed: {e}"


def build_agent_tools(scanned_drug: Optional[str] = None) -> list:
    """Build LangChain ``Tool`` list with scanned-drug context."""
    from langchain.agents import Tool

    sd = scanned_drug or ""

    def _interaction(q: str) -> str:
        return interaction_check(q, scanned_drug=sd or None)

    def _adverse(q: str) -> str:
        return adverse_events(q, scanned_drug=sd or None)

    def _vector(q: str) -> str:
        return vector_search(q, scanned_drug=sd or None)

    return [
        Tool(
            name="vector_search",
            func=_vector,
            description=(
                "Search FDA label chunks with vector DB + rerankers. "
                "Use for dosage, warnings, ingredients, directions. "
                "Input: the user's question."
            ),
        ),
        Tool(
            name="interaction_check",
            func=_interaction,
            description=(
                "Check drug-drug interactions (DDInter + reranked label chunks). "
                "Input: other drug name or full question. Scanned drug is known."
            ),
        ),
        Tool(
            name="adverse_events",
            func=_adverse,
            description=(
                "Side effects: FAERS + reranked adverse_reactions chunks. "
                "Input: question or drug name."
            ),
        ),
    ]
