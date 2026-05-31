"""
brain.py — Unified MedLabel pipeline.

Architecture
------------
                        User Question
                             │
                             ▼
                    ┌─────────────────┐
                    │   Grok Router   │  classifies intent into one of 3 functions
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       vector_search  interaction_check  adverse_events
              │              │              │
              └──────────────┴──────────────┘
                             │
                    ChromaDB (BGE-M3 embeddings)
                             │
                    Two-stage reranker
                    (BGE reranker → cross-encoder)
                             │
                    Top-K chunks
                             │
                    Grok synthesizes final answer
                             │
                    QueryResult returned to caller

Requires: XAI_API_KEY in .env
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GROK_MODEL    = "grok-3"
GROK_BASE_URL = "https://api.x.ai/v1"
TOP_K_FINAL   = 5

# ── Grok client singleton ─────────────────────────────────────────────────────
_grok_client = None

def _get_grok():
    global _grok_client
    if _grok_client is None:
        from openai import OpenAI
        key = os.getenv("XAI_API_KEY", "")
        if not key:
            raise ValueError(
                "XAI_API_KEY not found. Add XAI_API_KEY=your_key to .env\n"
                "Get a key at: https://console.x.ai/"
            )
        _grok_client = OpenAI(api_key=key, base_url=GROK_BASE_URL)
    return _grok_client


# ── Result dataclasses ────────────────────────────────────────────────────────
@dataclass
class Hit:
    text:    str   = ""
    score:   float = 0.0
    section: str   = ""
    source:  str   = ""

@dataclass
class QueryResult:
    function_used: str       = ""
    drug_name:     str       = ""
    query:         str       = ""
    hits:          list[Hit] = field(default_factory=list)
    hit_count:     int       = 0
    summary:       str       = ""


# ── Intent router ─────────────────────────────────────────────────────────────
_ROUTER_SYSTEM = """
You are a medical query router for a medicine label reading app.
Given a user question, respond with EXACTLY one of these three function names — nothing else:
  vector_search      → dosage, warnings, ingredients, directions, how to use,
                        active ingredient, indications, overdose, pregnancy
  interaction_check  → drug interactions, can I take this with X, combining
                        medications, is it safe with another drug
  adverse_events     → side effects, adverse events, reported reactions,
                        FDA reports, how many people had problems, risks
Reply with only the function name. No punctuation, no explanation.
""".strip()

def _route_query(query: str) -> str:
    client = _get_grok()
    resp = client.chat.completions.create(
        model=GROK_MODEL,
        messages=[
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user",   "content": query},
        ],
        temperature=0,
        max_tokens=10,
    )
    intent = resp.choices[0].message.content.strip().lower()
    for name in ("vector_search", "interaction_check", "adverse_events"):
        if name in intent:
            logger.info("Routed '%s' → %s", query[:60], name)
            return name
    logger.warning("Unexpected routing '%s', defaulting to vector_search", intent)
    return "vector_search"


# ── Grok answer synthesis ─────────────────────────────────────────────────────
_ANSWER_SYSTEM = """
You are a medical label assistant. The user asked a question about a medicine.
You are given relevant excerpts from the FDA drug label retrieved from a vector database.
Answer the question clearly and concisely based ONLY on the provided context.
If the context does not contain enough information, say so.
Do not add information not present in the context.
""".strip()

def _grok_answer(query: str, chunks: list[dict], drug_name: str) -> str:
    if not chunks:
        return f"No relevant information found in the database for '{drug_name}'."

    context_parts = []
    for i, c in enumerate(chunks, 1):
        section = c.get("metadata", {}).get("section_type", "label")
        score   = c.get("final_score", 0)
        context_parts.append(
            f"[{i}] (section: {section}, relevance: {score:.3f})\n{c['text']}"
        )

    context = "\n\n".join(context_parts)
    client  = _get_grok()

    resp = client.chat.completions.create(
        model=GROK_MODEL,
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user",   "content": (
                f"Drug: {drug_name}\n\n"
                f"Context from FDA label (retrieved from ChromaDB):\n{context}\n\n"
                f"Question: {query}"
            )},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


def _chunks_to_hits(chunks: list[dict]) -> list[Hit]:
    return [
        Hit(
            text    = c["text"],
            score   = c.get("final_score", 0.0),
            section = c.get("metadata", {}).get("section_type", ""),
            source  = c.get("metadata", {}).get("source", ""),
        )
        for c in chunks[:TOP_K_FINAL]
    ]


# ── Drug name extraction (fallback if not provided) ───────────────────────────
_STOP = {"can", "i", "take", "with", "is", "it", "safe", "to", "what",
         "are", "the", "does", "do", "how", "much", "a", "an", "this",
         "drug", "medication", "medicine", "pill", "for", "of", "and"}

def _extract_drug_name(query: str) -> str:
    tokens = re.findall(r"[A-Za-z][a-zA-Z0-9\-]+", query)
    names  = [t for t in tokens if t.lower() not in _STOP and len(t) >= 3]
    upper  = [n for n in names if n[0].isupper()]
    candidates = upper or names
    return candidates[0] if candidates else ""


# ── The three query functions (ChromaDB-backed) ───────────────────────────────
def vector_search(query: str, drug_name: str = "") -> QueryResult:
    from knowledge.query_functions import vector_search as chroma_vs

    if not drug_name:
        drug_name = _extract_drug_name(query)

    logger.info("[vector_search] drug=%r  query=%r", drug_name, query)

    chunks = chroma_vs(
        query,
        drug_name=drug_name if drug_name else None,
        top_n=TOP_K_FINAL,
    )

    if not chunks:
        return QueryResult(
            function_used="vector_search",
            drug_name=drug_name,
            query=query,
            summary=f"No information found in the database for '{drug_name}'. "
                    "Please verify the drug name is in our database.",
        )

    summary = _grok_answer(query, chunks, drug_name)
    hits    = _chunks_to_hits(chunks)

    return QueryResult(
        function_used = "vector_search",
        drug_name     = drug_name,
        query         = query,
        hits          = hits,
        hit_count     = len(hits),
        summary       = summary,
    )


def interaction_check(query: str, drug_name: str = "") -> QueryResult:
    from knowledge.query_functions import interaction_check as chroma_ic

    tokens     = re.findall(r"[A-Za-z][a-zA-Z0-9\-]+", query)
    candidates = [t for t in tokens if t.lower() not in _STOP and len(t) >= 3]
    if drug_name and drug_name not in candidates:
        candidates.insert(0, drug_name)

    drug_a = candidates[0] if len(candidates) > 0 else drug_name
    drug_b = candidates[1] if len(candidates) > 1 else None
    label  = f"{drug_a} + {drug_b}" if drug_b else drug_a

    logger.info("[interaction_check] drug_a=%r drug_b=%r  query=%r", drug_a, drug_b, query)

    chunks = chroma_ic(drug_a, drug_b, top_n=TOP_K_FINAL)

    if not chunks:
        return QueryResult(
            function_used="interaction_check",
            drug_name=label,
            query=query,
            summary=f"No interaction data found for '{label}' in the database. "
                    "Consult your pharmacist.",
        )

    summary = _grok_answer(query, chunks, label)
    hits    = _chunks_to_hits(chunks)

    return QueryResult(
        function_used = "interaction_check",
        drug_name     = label,
        query         = query,
        hits          = hits,
        hit_count     = len(hits),
        summary       = summary,
    )


def adverse_events(query: str, drug_name: str = "") -> QueryResult:
    from knowledge.query_functions import adverse_events as chroma_ae

    if not drug_name:
        drug_name = _extract_drug_name(query)

    logger.info("[adverse_events] drug=%r  query=%r", drug_name, query)

    chunks = chroma_ae(drug_name, top_n=TOP_K_FINAL)

    if not chunks:
        return QueryResult(
            function_used="adverse_events",
            drug_name=drug_name,
            query=query,
            summary=f"No adverse event data found for '{drug_name}' in the database.",
        )

    summary = _grok_answer(query, chunks, drug_name)
    hits    = _chunks_to_hits(chunks)

    return QueryResult(
        function_used = "adverse_events",
        drug_name     = drug_name,
        query         = query,
        hits          = hits,
        hit_count     = len(hits),
        summary       = summary,
    )


# ── MedBrain — public entry point ─────────────────────────────────────────────
class MedBrain:
    """
    Main agent. Routes question → ChromaDB retrieval → Grok answer synthesis.
    Requires XAI_API_KEY in .env.
    """
    FUNCTION_MAP = {
        "vector_search":     vector_search,
        "interaction_check": interaction_check,
        "adverse_events":    adverse_events,
    }

    def ask(self, question: str, drug_name: str = "") -> QueryResult:
        intent = _route_query(question)
        fn     = self.FUNCTION_MAP[intent]
        return fn(question, drug_name=drug_name)

    def ask_all(self, question: str, drug_name: str = "") -> dict[str, QueryResult]:
        return {
            name: fn(question, drug_name=drug_name)
            for name, fn in self.FUNCTION_MAP.items()
        }


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    brain = MedBrain()

    test_cases = [
        ("What is the dosage for warfarin?",                     "warfarin"),
        ("Can I take aspirin with warfarin?",                    "warfarin"),
        ("What are the most reported side effects of warfarin?", "warfarin"),
    ]

    for question, drug in test_cases:
        print(f"\n{'─'*60}")
        print(f"Q: {question}")
        result = brain.ask(question, drug_name=drug)
        print(f"  Routed to  : {result.function_used}")
        print(f"  Drug       : {result.drug_name}")
        print(f"  Hits found : {result.hit_count}  (from ChromaDB)")
        print(f"  Answer     : {result.summary[:300]}")