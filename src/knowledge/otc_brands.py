"""
Common OTC products: openFDA fetch names, Chroma drug_name, and ingredient fallbacks.
"""

from __future__ import annotations

from typing import Optional

# (openFDA search term, canonical drug_name stored in Chroma / shown in UI)
OTC_BRAND_LABELS: list[tuple[str, str]] = [
    ("NYQUIL", "NyQuil"),
    ("DAYQUIL", "DayQuil"),
    ("TYLENOL", "Tylenol"),
    ("ADVIL", "Advil"),
    ("MOTRIN", "Motrin"),
    ("ALEVE", "Aleve"),
    ("BENADRYL", "Benadryl"),
    ("CLARITIN", "Claritin"),
    ("ZYRTEC", "Zyrtec"),
    ("ROBITUSSIN", "Robitussin"),
    ("MUCINEX", "Mucinex"),
    ("PRILOSEC", "Prilosec"),
    ("PEPTO-BISMOL", "Pepto-Bismol"),
    ("FLONASE", "Flonase"),
    ("SUDAFED", "Sudafed"),
    ("BAYER ASPIRIN", "Bayer Aspirin"),
]

# If no brand label in DB yet, search these ingredient generics (openFDA names).
OTC_BRAND_INGREDIENTS: dict[str, list[str]] = {
    "nyquil": ["acetaminophen", "dextromethorphan", "doxylamine"],
    "dayquil": ["acetaminophen", "dextromethorphan", "phenylephrine"],
    "tylenol": ["acetaminophen"],
    "advil": ["ibuprofen"],
    "motrin": ["ibuprofen"],
    "aleve": ["naproxen"],
    "benadryl": ["diphenhydramine"],
    "claritin": ["loratadine"],
    "zyrtec": ["cetirizine"],
    "robitussin": ["dextromethorphan"],
    "mucinex": ["guaifenesin"],
    "prilosec": ["omeprazole"],
    "pepto-bismol": ["bismuth subsalicylate"],
    "flonase": ["fluticasone propionate"],
    "sudafed": ["pseudoephedrine"],
    "bayer aspirin": ["aspirin"],
}

CANONICAL_BRANDS: set[str] = {display for _, display in OTC_BRAND_LABELS}


def canonical_brand_for_query(brand: str) -> Optional[str]:
    """Return canonical display name (e.g. NyQuil) if brand is a known OTC product."""
    key = (brand or "").strip().lower()
    if not key:
        return None
    for query, display in OTC_BRAND_LABELS:
        if display.lower() == key or query.lower() == key:
            return display
    return None


def ingredient_tokens_for_brand(brand: str) -> list[str]:
    """Search tokens: canonical brand first, then ingredient fallbacks."""
    key = (brand or "").strip().lower()
    tokens: list[str] = []
    if not key:
        return tokens

    for _query, display in OTC_BRAND_LABELS:
        if display.lower() == key or _query.lower() == key:
            tokens.append(display)
            break
    else:
        tokens.append(key)

    if key in OTC_BRAND_INGREDIENTS:
        for ing in OTC_BRAND_INGREDIENTS[key]:
            if ing not in tokens:
                tokens.append(ing)
    return tokens
