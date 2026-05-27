"""
LangChain agent tools for MedLabel.

Team ownership:
  - vector_search      — ChromaDB + BGE-M3 rerank pipeline
  - interaction_check  — DDInter + FDA (implemented in interaction_check.py)
  - adverse_events     — openFDA /drug/event
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from agent.interaction_check import interaction_check as run_interaction_check

if TYPE_CHECKING:
    from langchain.agents import Tool
from api.openfda import OpenFDAClient


def vector_search(query: str) -> str:
    """Placeholder until ChromaDB + rerank pipeline is wired (teammate)."""
    return (
        "vector_search is not wired yet. "
        "Use interaction_check for drug–drug questions or adverse_events for side effects."
    )


def adverse_events(drug_name: str) -> str:
    """
    Top reported adverse reactions from openFDA FAERS (drug/event).

    ``drug_name`` should be generic or brand name from the scanned label.
    """
    name = (drug_name or "").strip()
    if not name:
        return "adverse_events requires a drug name."
    client = OpenFDAClient()
    events = client.get_adverse_events(name, limit=10)
    if not events:
        return f"No adverse event summary found in openFDA for '{name}'."
    lines = [f"Top reported adverse events for {name} (openFDA FAERS):", ""]
    for row in events:
        lines.append(f"  - {row.get('reaction', '?')}: {row.get('count', 0)} reports")
    lines.append(
        "\n⚠️ FAERS reports are voluntary and do not prove causation. Consult a healthcare professional."
    )
    return "\n".join(lines)


def interaction_check(query: str, scanned_drug: Optional[str] = None) -> str:
    """
    Check drug–drug interactions via DDInter + FDA label interactions section.

    Parameters
    ----------
    query:
        Other drug (e.g. ``ibuprofen``) or pair ``acetaminophen|ibuprofen``.
    scanned_drug:
        Drug from the scanned label (first drug in the pair).
    """
    try:
        result = run_interaction_check(query, scanned_drug=scanned_drug)
        return result.format_for_agent()
    except ValueError as e:
        return f"interaction_check error: {e}"
    except Exception as e:
        return f"interaction_check failed: {e}"


def build_agent_tools(scanned_drug: Optional[str] = None) -> list:
    """
    Build LangChain ``Tool`` list with optional scanned-drug context for interactions.

    Example::
        tools = build_agent_tools(scanned_drug="acetaminophen")
    """
    from langchain.agents import Tool

    sd = scanned_drug or ""

    def _interaction(q: str) -> str:
        return interaction_check(q, scanned_drug=sd or None)

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
                "Input: the other drug name (e.g. ibuprofen), or drug_a|drug_b. "
                "The scanned medicine is already known from context."
            ),
        ),
        Tool(
            name="adverse_events",
            func=adverse_events,
            description=(
                "Get commonly reported side effects from FDA adverse event reports. "
                "Input: drug name (generic or brand)."
            ),
        ),
    ]
