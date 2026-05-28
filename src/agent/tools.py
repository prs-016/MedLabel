"""
LangChain agent tools for MedLabel.

Team ownership:
  - vector_search      — ChromaDB + BGE-M3 rerank pipeline
  - interaction_check  — DDInter + FDA (interaction_check.py)
  - adverse_events     — openFDA FAERS + label (adverse_events.py)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from agent.adverse_events import lookup_adverse_events
from agent.interaction_check import interaction_check as run_interaction_check

if TYPE_CHECKING:
    from langchain.agents import Tool


def vector_search(query: str) -> str:
    """Placeholder until ChromaDB + rerank pipeline is wired (teammate)."""
    return (
        "vector_search is not wired yet. "
        "Use interaction_check for drug–drug questions or adverse_events for side effects."
    )


def adverse_events(drug_name: str, scanned_drug: Optional[str] = None) -> str:
    """
    Top reported adverse reactions from openFDA FAERS + FDA label section.

    Parameters
    ----------
    drug_name:
        Drug name from the question, or pass scanned drug when user asks about "this" medicine.
    scanned_drug:
        Drug from the scanned label.
    """
    try:
        result = lookup_adverse_events(drug_name, scanned_drug=scanned_drug)
        return result.format_for_agent()
    except ValueError as e:
        return f"adverse_events error: {e}"
    except Exception as e:
        return f"adverse_events failed: {e}"


def interaction_check(query: str, scanned_drug: Optional[str] = None) -> str:
    """
    Check drug–drug interactions via DDInter + FDA label interactions section.
    """
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

    return [
        Tool(
            name="vector_search",
            func=vector_search,
            description=(
                "Search FDA label chunks in the vector database for dosage, warnings, "
                "ingredients, and directions. Input: the user's question."
            ),
        ),
        Tool(
            name="interaction_check",
            func=_interaction,
            description=(
                "Check if two drugs interact. Use when user asks 'Can I take this with X?'. "
                "Input: the other drug name (e.g. ibuprofen). "
                "The scanned medicine is already known from context."
            ),
        ),
        Tool(
            name="adverse_events",
            func=_adverse,
            description=(
                "Get commonly reported side effects from FDA adverse event reports and label. "
                "Input: drug name, or ask about side effects of the scanned medicine."
            ),
        ),
    ]
