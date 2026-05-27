"""
Image scan pipeline for MedLabel demos.

Upload / file path → ``run_ocr`` → RxNorm-normalized drug name for the agent.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from api.rxnorm import RxNormClient
from vision.ocr import OCRResult, run_ocr

logger = logging.getLogger(__name__)

PackagingType = Literal["flat", "cylindrical"]


def scan_image(image_path: str, packaging_type: PackagingType) -> OCRResult:
    """Run Path A (flat) or Path B (cylindrical) OCR on an image file."""
    return run_ocr(image_path, packaging_type)


def resolve_scanned_drug(ocr_result: OCRResult, rxnorm: RxNormClient | None = None) -> str:
    """
    Best-effort generic drug name from OCR output (RxNorm when possible).
    """
    rx = rxnorm or RxNormClient()
    candidate = (ocr_result.drug_name or "").strip()
    if not candidate and ocr_result.raw_text:
        for line in ocr_result.raw_text.splitlines():
            line = line.strip()
            if len(line) >= 3:
                candidate = line
                break
    if not candidate:
        return ""
    resolved = rx.resolve_ocr_string(candidate)
    return (
        resolved.generic_name
        or resolved.matched_name
        or candidate
    ).strip()


def ocr_result_to_dict(ocr_result: OCRResult, scanned_drug: str = "") -> dict[str, Any]:
    """JSON-serializable scan payload for Streamlit / APIs."""
    return {
        "drug_name": ocr_result.drug_name,
        "scanned_drug": scanned_drug or ocr_result.drug_name,
        "dosage": ocr_result.dosage,
        "active_ingredients": ocr_result.active_ingredients,
        "warnings": ocr_result.warnings,
        "directions": ocr_result.directions,
        "expiry_date": ocr_result.expiry_date,
        "path_used": ocr_result.path_used,
        "confidence": ocr_result.confidence,
        "hallucination_flags": list(ocr_result.hallucination_flags),
        "passed_hallucination_check": ocr_result.passed_hallucination_check,
        "raw_text_preview": (ocr_result.raw_text or "")[:2000],
    }
