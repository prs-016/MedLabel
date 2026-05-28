"""
brain.py — Intent router + retrieval pipeline for MedLabel.

Architecture
------------

                        User Question
                             │
                             ▼
                    ┌─────────────────┐
                    │   Grok Router   │  ← structured prompt classifies
                    │   (xAI API)     │    into one of 3 functions
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       vector_search  interaction_check  adverse_events
       (dosage /       (drug A + B       (FDA reported
        warnings /      interactions)     reactions)
        ingredients)
              │              │              │
              └──────────────┴──────────────┘
                             │
                    Shared Retrieval Pipeline
                    ─────────────────────────
                    1. Fetch chunks from OpenFDA
                    2. BM25 keyword rank → top K chunks
                    3. Send chunks to Grok as context
                    4. Grok synthesises structured answer
                    5. Return ranked hits

No local ML models — zero torch / paddle conflict.
Requires: XAI_API_KEY in .env   |   pip install openai rank-bm25
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

TOP_K_BM25    = 8    # chunks kept after BM25 pass
TOP_K_FINAL   = 5    # hits returned to caller


# ── Grok client singleton ─────────────────────────────────────────────────────

_grok_client = None

def _get_grok():
    global _grok_client
    if _grok_client is None:
        from openai import OpenAI
        key = os.getenv("XAI_API_KEY", "")
        if not key:
            raise ValueError(
                "XAI_API_KEY not found. Add XAI_API_KEY=your_key to your .env file.\n"
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
    """Ask Grok to classify the query into one of the three functions."""
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

    # Normalise in case Grok adds punctuation
    for name in ("vector_search", "interaction_check", "adverse_events"):
        if name in intent:
            logger.info("Routed '%s' → %s", query[:60], name)
            return name

    # Fallback
    logger.warning("Unexpected routing response '%s', defaulting to vector_search", intent)
    return "vector_search"


# ── BM25 retrieval ────────────────────────────────────────────────────────────

def _bm25_rank(query: str, chunks: list[dict], top_k: int = TOP_K_BM25) -> list[dict]:
    """
    Keyword-based BM25 ranking — no torch, no paddle, pure Python.
    Returns up to top_k chunks sorted by relevance score.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError(
            "rank-bm25 is not installed. Run: pip install rank-bm25"
        )

    tokenised = [c["text"].lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenised)
    scores    = bm25.get_scores(query.lower().split())

    ranked = sorted(
        zip(chunks, scores.tolist()), key=lambda x: x[1], reverse=True
    )
    return [c for c, s in ranked[:top_k] if s > 0] or [c for c, _ in ranked[:top_k]]


# ── Grok answer synthesis ─────────────────────────────────────────────────────

_ANSWER_SYSTEM = """
You are a medical label assistant. The user asked a question about a medicine.
You are given relevant excerpts from the FDA drug label or adverse event database.
Answer the question clearly and concisely based ONLY on the provided context.
If the context does not contain enough information, say so.
Do not add information not present in the context.
""".strip()


def _grok_answer(query: str, chunks: list[dict], drug_name: str) -> str:
    """Send top chunks to Grok as context and get a synthesised answer."""
    if not chunks:
        return f"No relevant information found for '{drug_name}'."

    context_parts = []
    for i, c in enumerate(chunks, 1):
        section = c.get("metadata", {}).get("section_type", "label")
        context_parts.append(f"[{i}] ({section})\n{c['text']}")
    context = "\n\n".join(context_parts)

    client = _get_grok()
    resp = client.chat.completions.create(
        model=GROK_MODEL,
        messages=[
            {"role": "system",  "content": _ANSWER_SYSTEM},
            {"role": "user",    "content": (
                f"Drug: {drug_name}\n\n"
                f"Context from FDA label:\n{context}\n\n"
                f"Question: {query}"
            )},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


def _chunks_to_hits(chunks: list[dict], scores: list[float] | None = None) -> list[Hit]:
    """Convert ranked chunks into Hit objects."""
    hits = []
    for i, c in enumerate(chunks[:TOP_K_FINAL]):
        hits.append(Hit(
            text    = c["text"],
            score   = float(scores[i]) if scores else float(TOP_K_FINAL - i),
            section = c.get("metadata", {}).get("section_type", ""),
            source  = c.get("metadata", {}).get("source", ""),
        ))
    return hits


# ── Drug name extraction ──────────────────────────────────────────────────────

_STOP = {"can", "i", "take", "with", "is", "it", "safe", "to", "what",
         "are", "the", "does", "do", "how", "much", "a", "an", "this",
         "drug", "medication", "medicine", "pill", "for", "of", "and"}

def _extract_drug_names(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][a-zA-Z0-9\-]+", query)
    names  = [t for t in tokens if t.lower() not in _STOP and len(t) >= 3]
    upper  = [n for n in names if n[0].isupper()]
    return (upper or names)[:2]


# ── The three query functions ─────────────────────────────────────────────────

def vector_search(query: str, drug_name: str = "") -> QueryResult:
    """
    General label search — dosage, warnings, ingredients, directions.
    Fetches all FDA label sections, BM25 ranks them, Grok answers.
    """
    from api.openfda import OpenFDAClient

    if not drug_name:
        candidates = _extract_drug_names(query)
        drug_name  = candidates[0] if candidates else ""

    logger.info("[vector_search] drug=%r  query=%r", drug_name, query)

    client = OpenFDAClient()
    chunks = client.get_label_chunks(drug_name) if drug_name else []

    if not chunks:
        return QueryResult(
            function_used="vector_search",
            drug_name=drug_name,
            query=query,
            summary=f"No FDA label found for '{drug_name}'. Please verify the drug name.",
        )

    top_chunks = _bm25_rank(query, chunks)
    summary    = _grok_answer(query, top_chunks, drug_name)
    hits       = _chunks_to_hits(top_chunks)

    return QueryResult(
        function_used = "vector_search",
        drug_name     = drug_name,
        query         = query,
        hits          = hits,
        hit_count     = len(hits),
        summary       = summary,
    )


def interaction_check(query: str, drug_name: str = "") -> QueryResult:
    """
    Drug interaction check.
    Fetches interaction sections from both drugs' FDA labels, Grok answers.
    """
    from api.openfda import OpenFDAClient

    candidates = _extract_drug_names(query)
    if drug_name and drug_name not in candidates:
        candidates.insert(0, drug_name)
    candidates = candidates[:2]

    logger.info("[interaction_check] drugs=%r  query=%r", candidates, query)

    INTERACTION_SECTIONS = {
        "drug_interactions", "ask_doctor_or_pharmacist",
        "warnings", "ask_doctor",
    }

    client = OpenFDAClient()
    chunks: list[dict] = []

    for name in candidates:
        label = client.get_label(name)
        if label:
            for chunk in label.to_chunks():
                if chunk["metadata"].get("section_type") in INTERACTION_SECTIONS:
                    chunks.append(chunk)

    drug_label = " + ".join(candidates) if candidates else drug_name

    if not chunks:
        return QueryResult(
            function_used="interaction_check",
            drug_name=drug_label,
            query=query,
            summary=f"No interaction data found for '{drug_label}'. Consult your pharmacist.",
        )

    top_chunks = _bm25_rank(query, chunks)
    summary    = _grok_answer(query, top_chunks, drug_label)
    hits       = _chunks_to_hits(top_chunks)

    return QueryResult(
        function_used = "interaction_check",
        drug_name     = drug_label,
        query         = query,
        hits          = hits,
        hit_count     = len(hits),
        summary       = summary,
    )


def adverse_events(query: str, drug_name: str = "") -> QueryResult:
    """
    FDA adverse event lookup.
    Pulls top reported reactions, BM25 ranks them, Grok answers.
    """
    from api.openfda import OpenFDAClient

    if not drug_name:
        candidates = _extract_drug_names(query)
        drug_name  = candidates[0] if candidates else ""

    logger.info("[adverse_events] drug=%r  query=%r", drug_name, query)

    client = OpenFDAClient()
    events = client.get_adverse_events(drug_name, limit=50) if drug_name else []

    if not events:
        return QueryResult(
            function_used="adverse_events",
            drug_name=drug_name,
            query=query,
            summary=f"No adverse event data found for '{drug_name}'.",
        )

    # Convert event list → chunks (10 reactions per chunk)
    chunks: list[dict] = []
    for i in range(0, len(events), 10):
        batch = events[i : i + 10]
        text  = "; ".join(
            f"{e['reaction']} ({e['count']} reports)" for e in batch
        )
        chunks.append({
            "text": text,
            "metadata": {
                "drug_name":    drug_name,
                "section_type": "adverse_events",
                "source":       "fda_adverse_events",
            },
        })

    top_chunks = _bm25_rank(query, chunks)
    summary    = _grok_answer(query, top_chunks, drug_name)
    hits       = _chunks_to_hits(top_chunks)

    # Also surface the raw top-5 events sorted by count
    top_events   = sorted(events, key=lambda e: e["count"], reverse=True)[:5]
    top_summary  = "Most reported: " + ", ".join(
        f"{e['reaction']} ({e['count']})" for e in top_events
    )
    summary = f"{summary}\n\n{top_summary}"

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
    Main agent. Call ask(question, drug_name) — Grok routes it to the right
    function, BM25 retrieves relevant FDA chunks, Grok synthesises the answer.

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
        """Run all three functions — useful for debugging."""
        return {
            name: fn(question, drug_name=drug_name)
            for name, fn in self.FUNCTION_MAP.items()
        }


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    brain = MedBrain()

    test_cases = [
        ("What is the dosage for Claritin?",                 "Claritin"),
        ("Can I take Advil with Tylenol?",                   "Advil"),
        ("What are the most reported side effects of Advil?", "Advil"),
    ]

    for question, drug in test_cases:
        print(f"\n{'─'*60}")
        print(f"Q: {question}")
        result = brain.ask(question, drug_name=drug)
        print(f"  Routed to  : {result.function_used}")
        print(f"  Drug       : {result.drug_name}")
        print(f"  Hits found : {result.hit_count}")
        print(f"  Answer     : {result.summary[:300]}")
