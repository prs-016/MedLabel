"""
src/agent/simplifier.py
========================
Gemini-powered plain-English label simplifier.

Takes an OCRResult and returns a summary at a 5th-grade reading level.
Requires GEMINI_API_KEY in .env.
"""

from __future__ import annotations

import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

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


def simplify_label(ocr_result) -> str:
    """
    Simplify an OCRResult into plain English using Gemini 2.5 Flash.

    Returns the formatted summary string, or an explanatory error message
    if the API key is missing or the call fails.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return (
            "Gemini API key is not configured. "
            "Add `GEMINI_API_KEY=your_key` to your .env file."
        )

    label_text = _build_label_text(ocr_result)
    if not label_text.strip():
        return "No label text available to simplify."

    try:
        return _call_gemini(label_text, api_key)
    except Exception as exc:
        logger.error("Gemini simplify error: %s", exc)
        return f"Could not generate summary: {exc}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_label_text(ocr_result) -> str:
    """Assemble the extracted fields into a single text block for Gemini."""
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
    # Fall back to raw text when structured extraction found nothing
    if not parts and ocr_result.raw_text:
        parts.append(f"Label text:\n{ocr_result.raw_text[:3000]}")
    return "\n".join(parts)


def _call_gemini(label_text: str, api_key: str) -> str:
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": _SYSTEM_PROMPT + "\n\nLabel information:\n" + label_text}
                ]
            }
        ]
    }

    resp = requests.post(
        _GEMINI_URL,
        params={"key": api_key},
        json=payload,
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Gemini API returned {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response shape: {exc}")
