"""
src/agent/simplifier.py
========================
Grok-powered plain-English label simplifier.

Takes an OCRResult and returns a summary at a 5th-grade reading level.
Requires XAI_API_KEY in .env (same key as the chatbot).
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GROK_MODEL = os.getenv("XAI_TEXT_MODEL", "grok-3")
GROK_BASE_URL = "https://api.x.ai/v1"

_SYSTEM_PROMPT = """\
You are a medical label assistant helping everyday people understand medicine labels.
Simplify the following label information into clear, plain English at a 5th-grade reading level.

Format your response exactly like this (use markdown bold headers):
**What this medicine is:** (1-2 sentences)
**How to take it:** (short bullet points)
**Important warnings:** (bullet points, max 3 most critical)
**When it expires:** (if available, otherwise omit this section)

Rules:
- Use simple words — no medical jargon.
- Keep each bullet point to one short sentence.
- If a section has no information, omit it entirely.
- End your response with this exact line:
  ⚠️ Always verify with the physical label or your pharmacist.
"""

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        key = os.getenv("XAI_API_KEY", "")
        if not key:
            raise ValueError(
                "XAI_API_KEY not set. Add it to your .env file for plain-English summaries."
            )
        _client = OpenAI(api_key=key, base_url=GROK_BASE_URL)
    return _client


def simplify_label(ocr_result) -> str:
    """
    Simplify an OCRResult into plain English using Grok.

    Returns the formatted summary string, or an explanatory error message
    if the API key is missing or the call fails.
    """
    if not os.getenv("XAI_API_KEY", ""):
        return (
            "xAI API key is not configured. "
            "Add `XAI_API_KEY=your_key` to your .env file."
        )

    label_text = _build_label_text(ocr_result)
    if not label_text.strip():
        return "No label text available to simplify."

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Label information:\n{label_text}",
                },
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error("Grok simplify error: %s", exc)
        return f"Could not generate summary: {exc}"


def _build_label_text(ocr_result) -> str:
    """Assemble the extracted fields into a single text block for the model."""
    parts: list[str] = []
    if ocr_result.drug_name:
        parts.append(f"Drug name: {ocr_result.drug_name}")
    if ocr_result.dosage:
        parts.append(f"Dosage: {ocr_result.dosage}")
    if ocr_result.active_ingredients:
        parts.append(f"Active ingredients: {ocr_result.active_ingredients}")
    if ocr_result.warnings:
        parts.append(f"Warnings: {ocr_result.warnings}")
    if ocr_result.directions:
        parts.append(f"Directions: {ocr_result.directions}")
    if ocr_result.expiry_date:
        parts.append(f"Expiry: {ocr_result.expiry_date}")
    if not parts and ocr_result.raw_text:
        parts.append(f"Label text:\n{ocr_result.raw_text[:3000]}")
    return "\n".join(parts)
