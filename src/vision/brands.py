"""
Known OTC / Rx product names for OCR correction after PaddleOCR.
"""

from __future__ import annotations

import re
from typing import Optional

# DayQuil before NyQuil in list; DayQuil gets higher priority when both could match.
_BRAND_MATCHERS: list[tuple[str, re.Pattern[str], int]] = [
    (
        "DayQuil",
        re.compile(r"\b(?:d[a4]y\s*qu[i1l][lj]?|dayqu[i1l])\b", re.I),
        22,
    ),
    (
        "NyQuil",
        re.compile(r"\b(?:[o0]?ny\s*qu[i1l][lj]?|nyqu[i1l])\b", re.I),
        21,
    ),
    ("Advil", re.compile(r"\badv[i1]l\b", re.I), 18),
    ("Motrin", re.compile(r"\bmotr[i1]n\b", re.I), 18),
    ("Tylenol", re.compile(r"\btylen[o0]l\b", re.I), 18),
    ("Aleve", re.compile(r"\baleve\b", re.I), 18),
    ("Benadryl", re.compile(r"\bbenadryl\b", re.I), 17),
    ("Claritin", re.compile(r"\bclaritin\b", re.I), 17),
    ("Zyrtec", re.compile(r"\bzyrtec\b", re.I), 17),
    ("Mucinex", re.compile(r"\bmucinex\b", re.I), 17),
    ("Robitussin", re.compile(r"\brobitussin\b", re.I), 17),
    ("Coumadin", re.compile(r"\bcoumadin\b", re.I), 15),
    ("Lipitor", re.compile(r"\blipitor\b", re.I), 15),
    ("Prilosec", re.compile(r"\bprilosec\b", re.I), 15),
    ("Glucophage", re.compile(r"\bglucophage\b", re.I), 15),
    ("Vicks", re.compile(r"\b[jv1][i1l]cks\b", re.I), 1),
]

_COLD_FLU = re.compile(r"cold\s*[&+and]*\s*flu", re.I)
_DAYQUIL_HINTS = re.compile(
    r"day\s*qu|dayquil|daytime|non[- ]?drowsy|phenylephrine",
    re.I,
)
_NYQUIL_HINTS = re.compile(
    r"(?<![a-z])ny\s*qu|nyquil|nighttime|doxylamine",
    re.I,
)

_GARBAGE_LINE = re.compile(
    r"medicine\s+so\s+you\s+can|rest\s+medicine|toucan",
    re.I,
)


def _all_brand_hits(text: str) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    for name, pattern, priority in _BRAND_MATCHERS:
        if pattern.search(text):
            hits.append((name, priority))
    return hits


def _disambiguate_day_vs_ny(full_text: str, names: set[str]) -> Optional[str]:
    """DayQuil and NyQuil share Vicks branding — use label cues, not bare 'quil'."""
    if "DayQuil" in names and "NyQuil" not in names:
        return "DayQuil"
    if "NyQuil" in names and "DayQuil" not in names:
        return "NyQuil"
    if "DayQuil" not in names and "NyQuil" not in names:
        return None

    if _DAYQUIL_HINTS.search(full_text):
        return "DayQuil"
    if _NYQUIL_HINTS.search(full_text):
        return "NyQuil"
    # Both name tokens on label — prefer explicit DayQuil token
    if "DayQuil" in names:
        return "DayQuil"
    return "NyQuil"


def _infer_from_label_context(text: str) -> Optional[str]:
    """Only infer when label cues are specific (not shared cold/flu boilerplate)."""
    if _DAYQUIL_HINTS.search(text):
        return "DayQuil"
    if _NYQUIL_HINTS.search(text):
        return "NyQuil"
    # Doxylamine is in NyQuil, not standard DayQuil
    if re.search(r"doxylamine", text, re.I):
        return "NyQuil"
    if re.search(r"phenylephrine", text, re.I) and _COLD_FLU.search(text):
        return "DayQuil"
    return None


def _pick_best_brand(hits: list[tuple[str, int]], full_text: str) -> str:
    if not hits:
        return _infer_from_label_context(full_text) or ""

    names = {h[0] for h in hits}
    if "DayQuil" in names or "NyQuil" in names:
        resolved = _disambiguate_day_vs_ny(full_text, names)
        if resolved:
            return resolved

    hits.sort(key=lambda x: (-x[1], x[0]))
    best_name = hits[0][0]

    if best_name == "Vicks":
        inferred = _infer_from_label_context(full_text)
        if inferred:
            return inferred
        if "DayQuil" in names:
            return "DayQuil"
        if "NyQuil" in names:
            return "NyQuil"

    return best_name


def match_brand_in_text(text: str) -> Optional[tuple[str, int]]:
    hits = _all_brand_hits(text)
    if not hits:
        return None
    hits.sort(key=lambda x: (-x[1], x[0]))
    return hits[0]


def extract_product_name_from_ocr_lines(lines: list[dict]) -> str:
    full_text = "\n".join(ln.get("text", "") for ln in lines)

    doc_hits = _all_brand_hits(full_text)
    if doc_hits:
        return _pick_best_brand(doc_hits, full_text)

    inferred = _infer_from_label_context(full_text)
    if inferred:
        return inferred

    line_hits: list[tuple[str, int]] = []
    for ln in lines:
        text = ln.get("text", "").strip()
        if not text or _GARBAGE_LINE.search(text):
            continue
        line_hits.extend(_all_brand_hits(text))

    if line_hits:
        return _pick_best_brand(line_hits, full_text)

    for ln in lines:
        text = ln.get("text", "").strip()
        if (
            ln.get("confidence", 0) >= 0.80
            and 3 <= len(text) <= 40
            and not _GARBAGE_LINE.search(text)
            and re.search(r"[a-zA-Z]{3,}", text)
        ):
            hit = match_brand_in_text(text)
            if hit and hit[0] != "Vicks":
                return hit[0]
            if hit and hit[0] == "Vicks":
                inferred = _infer_from_label_context(full_text)
                return inferred or hit[0]
            return text
    return ""
