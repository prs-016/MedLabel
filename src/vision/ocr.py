"""
OCR routes for MedLabel: Path A (PaddleOCR + OpenCV) and Path B (Gemini Vision).

Path A uses CLAHE + adaptive threshold, then PaddleOCR. Path B sends bottle photos
to Gemini and parses structured JSON. Both return ``OCRResult`` for downstream code.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from .orientation import (
    apply_rotation,
    auto_orient_image,
    estimate_rotation_degrees,
    refine_rotation_with_ocr_score,
)

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

GEMINI_PROMPT = """
You are reading the label on a curved medicine bottle.
Extract the following fields exactly as printed.
If a field is not visible, use an empty string.

Return ONLY a JSON object in this exact format, no other text:
{
  "drug_name": "",
  "dosage": "",
  "active_ingredients": "",
  "warnings": "",
  "directions": "",
  "expiry_date": ""
}
"""


# OpenCV 4.x Python uses ADAPTIVE_THRESH_GAUSSIAN_C; very old tutorials used ADAPTIVE_GAUSSIAN_C.
_CV_ADAPTIVE_GAUSSIAN = getattr(
    cv2, "ADAPTIVE_THRESH_GAUSSIAN_C", getattr(cv2, "ADAPTIVE_GAUSSIAN_C", 1)
)


@dataclass
class PathAOCRConfig:
    """Tunable OpenCV preprocessing for flat labels (Path A)."""

    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    adaptive_method: int = _CV_ADAPTIVE_GAUSSIAN
    threshold_type: int = cv2.THRESH_BINARY


def preprocess_flat_label(
    image_bgr: np.ndarray,
    cfg: PathAOCRConfig | None = None,
) -> np.ndarray:
    """Grayscale → CLAHE → adaptive threshold → BGR for PaddleOCR."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("preprocess_flat_label expects a BGR image (H, W, 3)")
    cfg = cfg or PathAOCRConfig()
    if cfg.adaptive_block_size % 2 == 0 or cfg.adaptive_block_size < 3:
        raise ValueError("adaptive_block_size must be an odd integer >= 3")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid)
    enhanced = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(
        enhanced,
        255,
        cfg.adaptive_method,
        cfg.threshold_type,
        cfg.adaptive_block_size,
        cfg.adaptive_c,
    )
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _normalize_paddle_result(raw: Any) -> list[dict[str, Any]]:
    """Convert PaddleOCR ``ocr()`` output into stable dict records."""
    if raw is None:
        return []
    first = raw[0] if isinstance(raw, (list, tuple)) and len(raw) > 0 else raw
    if not first:
        return []
    out: list[dict[str, Any]] = []
    for item in first:
        if not item or len(item) < 2:
            continue
        box, rec = item[0], item[1]
        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
            text, conf = rec[0], float(rec[1])
        else:
            continue
        out.append({"text": text, "confidence": conf, "bbox": box})
    return out


class PathAOCR:
    """PaddleOCR wrapper for Path A (English flat labels)."""

    def __init__(
        self,
        *,
        lang: str = "en",
        use_angle_cls: bool = True,
        show_log: bool = False,
        ocr_version: str | None = None,
    ) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore import-not-found
        except ImportError as e:
            missing_paddle = isinstance(e, ModuleNotFoundError) and e.name == "paddle"
            if not missing_paddle:
                missing_paddle = "no module named 'paddle'" in str(e).lower()
            extra = ""
            if missing_paddle:
                extra = (
                    " Install PaddlePaddle (`import paddle`). The PyPI package `paddleocr` does not install it. "
                    "Run: python -m pip install -r requirements-paddle.txt (same venv)."
                )
            raise ImportError(
                "Path A failed to import PaddleOCR/PaddlePaddle."
                + extra
                + f" Original error: {e!r}"
            ) from e

        kwargs: dict[str, Any] = dict(
            use_angle_cls=use_angle_cls,
            lang=lang,
            show_log=show_log,
        )
        if ocr_version is not None:
            kwargs["ocr_version"] = ocr_version
        try:
            self._ocr = PaddleOCR(**kwargs, use_gpu=False)
        except TypeError:
            self._ocr = PaddleOCR(**kwargs)
        self.use_angle_cls = use_angle_cls

    def ocr_raw(self, image_bgr: np.ndarray) -> Any:
        return self._ocr.ocr(image_bgr, cls=self.use_angle_cls)

    def run(
        self,
        image_bgr: np.ndarray,
        *,
        preprocess: bool = True,
        preprocess_cfg: PathAOCRConfig | None = None,
        auto_page_rotate: bool = False,
        page_rotate_mode: str = "heuristic",
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        if auto_page_rotate:
            if page_rotate_mode == "none":
                work = image_bgr.copy()
            elif page_rotate_mode == "ocr":
                angle = refine_rotation_with_ocr_score(image_bgr, lambda im: self.ocr_raw(im))
                work = apply_rotation(image_bgr, angle)
            else:
                work, _ = auto_orient_image(image_bgr)
        else:
            work = image_bgr.copy()

        ready = preprocess_flat_label(work, preprocess_cfg) if preprocess else work
        raw = self.ocr_raw(ready)
        lines = _normalize_paddle_result(raw)
        if min_confidence > 0:
            lines = [r for r in lines if r["confidence"] >= min_confidence]
        return lines


def run_path_a_ocr(
    image_bgr: np.ndarray,
    *,
    engine: PathAOCR | None = None,
    preprocess: bool = True,
    preprocess_cfg: PathAOCRConfig | None = None,
    auto_page_rotate: bool = False,
    page_rotate_mode: str = "heuristic",
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    ocr = engine or PathAOCR()
    return ocr.run(
        image_bgr,
        preprocess=preprocess,
        preprocess_cfg=preprocess_cfg,
        auto_page_rotate=auto_page_rotate,
        page_rotate_mode=page_rotate_mode,
        min_confidence=min_confidence,
    )


def estimate_page_rotation(image_bgr: np.ndarray) -> int:
    return estimate_rotation_degrees(image_bgr)


@dataclass
class OCRResult:
    """
    Structured output from either OCR path.
    Path A (PaddleOCR) and Path B (Gemini Vision) both return this
    so everything downstream works the same regardless of which path ran.
    """

    drug_name: str = ""
    dosage: str = ""
    active_ingredients: str = ""
    warnings: str = ""
    directions: str = ""
    expiry_date: str = ""
    raw_text: str = ""
    path_used: str = ""
    confidence: float = 0.0
    hallucination_flags: list[str] = field(default_factory=list)

    @property
    def passed_hallucination_check(self) -> bool:
        return len(self.hallucination_flags) == 0


def run_path_a(image_path: str) -> OCRResult:
    """
    Path A — PaddleOCR for flat labels (boxes, blister packs, cream tubes).
    Returns ``OCRResult`` with ``raw_text`` joined from detected lines; structured
    fields stay empty until a later extraction step (e.g. Gemini on text).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    engine = PathAOCR()
    lines = engine.run(img)
    raw_text = "\n".join(str(line["text"]) for line in lines)
    confs = [float(line["confidence"]) for line in lines]
    conf = sum(confs) / len(confs) if confs else 0.0
    return OCRResult(raw_text=raw_text, path_used="paddleocr", confidence=conf)


def run_path_b(image_path: str, api_key: Optional[str] = None) -> OCRResult:
    """
    Path B — Gemini Vision for cylindrical bottles.
    """
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError(
            "No Gemini API key found. "
            "Add GEMINI_API_KEY=your_key to your .env file."
        )

    image_b64, mime_type = _encode_image(image_path)
    raw_response = _call_gemini(image_b64, mime_type, key)
    result = _parse_gemini_response(raw_response)
    result = _check_hallucinations(result)

    return result


def _encode_image(image_path: str) -> tuple[str, str]:
    """Read image from disk, return (base64_string, mime_type)."""
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "heic": "image/heic",
        "heif": "image/heif",
    }
    mime_type = mime_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    return image_b64, mime_type


def _call_gemini(image_b64: str, mime_type: str, api_key: str) -> str:
    """Send image + prompt to Gemini 2.0 Flash, return raw text response."""
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_b64,
                        }
                    },
                    {"text": GEMINI_PROMPT},
                ]
            }
        ]
    }

    resp = requests.post(
        GEMINI_API_URL,
        params={"key": api_key},
        json=payload,
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Gemini API error {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response shape: {e}\n{data}")


def _parse_gemini_response(raw: str) -> OCRResult:
    """Parse Gemini's text response into an OCRResult."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().strip("`").strip()

    try:
        data = json.loads(cleaned)
        return OCRResult(
            drug_name=data.get("drug_name", "").strip(),
            dosage=data.get("dosage", "").strip(),
            active_ingredients=data.get("active_ingredients", "").strip(),
            warnings=data.get("warnings", "").strip(),
            directions=data.get("directions", "").strip(),
            expiry_date=data.get("expiry_date", "").strip(),
            raw_text=raw,
            path_used="gemini_vision",
            confidence=0.9,
        )
    except json.JSONDecodeError:
        logger.warning("Gemini returned non-JSON response: %s", raw[:200])
        return OCRResult(
            raw_text=raw,
            path_used="gemini_vision",
            confidence=0.3,
            hallucination_flags=["response_not_json"],
        )


def _check_hallucinations(result: OCRResult) -> OCRResult:
    """Sanity checks on what Gemini returned to catch hallucinated fields."""
    flags = list(result.hallucination_flags)

    if not result.drug_name.strip():
        flags.append("drug_name_empty")

    placeholder_pattern = re.compile(
        r"\bnot\s+available\b|\bn/?a\b|\bunknown\b|\bnot\s+visible\b"
        r"|\bnot\s+legible\b|\bcannot\s+read\b|\bunable\s+to\b",
        re.IGNORECASE,
    )
    for field_name, value in [
        ("drug_name", result.drug_name),
        ("dosage", result.dosage),
        ("active_ingredients", result.active_ingredients),
    ]:
        if value and placeholder_pattern.search(value):
            flags.append(f"placeholder_text_in_{field_name}")

    if result.dosage and not re.search(r"\d", result.dosage):
        flags.append("dosage_has_no_number")

    if result.expiry_date:
        date_pattern = re.compile(
            r"\d{1,4}[-/]\d{1,2}([-/]\d{1,4})?"
            r"|[A-Za-z]{3,9}\s+\d{4}"
            r"|\d{2}/\d{4}"
        )
        if not date_pattern.search(result.expiry_date):
            flags.append("expiry_date_invalid_format")

    for field_name, value in [
        ("drug_name", result.drug_name),
        ("active_ingredients", result.active_ingredients),
    ]:
        words = value.lower().split()
        for i in range(len(words) - 3):
            if words[i] == words[i + 1] == words[i + 2] == words[i + 3]:
                flags.append(f"repeated_gibberish_in_{field_name}")
                break

    if result.confidence < 0.5:
        flags.append("confidence_too_low")

    result.hallucination_flags = flags
    if flags:
        result.confidence = max(0.1, result.confidence - 0.2 * len(flags))
        logger.warning(
            "Hallucination flags for '%s': %s", result.drug_name, flags
        )

    return result


def run_ocr(image_path: str, packaging_type: str) -> OCRResult:
    """
    Main entry point called by the rest of the pipeline.

    packaging_type: "flat" → Path A (PaddleOCR); "cylindrical" → Path B (Gemini Vision).
    """
    if packaging_type == "cylindrical":
        return run_path_b(image_path)
    if packaging_type == "flat":
        return run_path_a(image_path)
    raise ValueError(
        f"Unknown packaging_type '{packaging_type}'. "
        "Expected 'flat' or 'cylindrical'."
    )
