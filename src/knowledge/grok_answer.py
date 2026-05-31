"""
Natural-language answers from reranked label chunks via xAI Grok.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GROK_MODEL    = os.getenv("XAI_TEXT_MODEL", "grok-3")
GROK_BASE_URL = "https://api.x.ai/v1"

_ANSWER_SYSTEM = """
You are a careful medical label assistant for patients and clinicians.
You are given excerpts from FDA drug labels (and sometimes interaction data).
Answer the user's question clearly in plain English using ONLY the provided context.

For questions about combining two medicines:
- If a drug-drug interaction entry is marked Unknown or has no management text,
  do NOT treat that as a confirmed harmful interaction. Say the database has
  no specific pair guidance and summarize relevant warnings from each drug's
  label excerpts (overdose limits, liver/GI/bleeding risks, duplicate ingredients).
- Never invent doses, contraindications, or interaction severity not in the context.

If the context is insufficient, say what is missing and suggest verifying with a pharmacist.
Keep answers concise (under 200 words unless the question needs more detail).
""".strip()

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.getenv("XAI_API_KEY", "")
        if not key:
            raise ValueError(
                "XAI_API_KEY not set. Add it to .env for natural-language answers."
            )
        _client = OpenAI(api_key=key, base_url=GROK_BASE_URL)
    return _client


def synthesize_answer(
    query: str,
    chunks: list[dict],
    *,
    drug_context: str = "",
    route_label: str = "",
) -> str:
    """
    Turn reranked chunks (query_functions format: text, metadata, scores)
    into a short Grok-written answer.
    """
    if not chunks:
        return (
            "I could not find relevant label text in the database for that question. "
            "Try rephrasing or ask about a drug in our seeded set."
        )

    context_parts = []
    for i, c in enumerate(chunks[:5], 1):
        meta = c.get("metadata") or {}
        section = meta.get("section_type", "label")
        drug = meta.get("drug_name", "")
        header = f"[{i}] {drug} — {section}" if drug else f"[{i}] ({section})"
        context_parts.append(f"{header}\n{c.get('text', '')[:1200]}")

    context = "\n\n".join(context_parts)
    drug_line = drug_context or "unspecified"
    route_line = f"\nSearch route: {route_label}" if route_label else ""

    client = _get_client()
    resp = client.chat.completions.create(
        model=GROK_MODEL,
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Primary drug context: {drug_line}{route_line}\n\n"
                    f"Label excerpts:\n{context}\n\n"
                    f"User question: {query}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return (resp.choices[0].message.content or "").strip()
