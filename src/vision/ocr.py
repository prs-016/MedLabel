import os
import re
import base64
import logging
import json
from dataclasses import dataclass, field
from typing import Optional, Literal

from dotenv import load_dotenv

from api.xai_client import require_xai_api_key, vision_completion

load_dotenv()

logger = logging.getLogger(__name__)

PATH_B_VISION_PROMPT = """
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


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class OCRResult:
    """
    Structured output from either OCR path.
    Path A (PaddleOCR) and Path B (xAI Vision) both return this
    so everything downstream works the same regardless of which path ran.
    """
    drug_name:            str = ""
    dosage:               str = ""
    active_ingredients:   str = ""
    warnings:             str = ""
    directions:           str = ""
    expiry_date:          str = ""
    raw_text:             str = ""
    path_used:            str = ""
    confidence:           float = 0.0
    hallucination_flags:  list[str] = field(default_factory=list)

    @property
    def passed_hallucination_check(self) -> bool:
        return len(self.hallucination_flags) == 0


# ── Path A — PaddleOCR ────────────────────────────────────────────────────────

_CONFIDENCE_THRESHOLD = 0.65

# Common dosage units to anchor extraction
_DOSAGE_UNITS = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|l|iu|%|mg/ml|mg/g|units?)\b",
    re.IGNORECASE,
)

# Patterns that typically mark the first line of a drug name block
_DRUG_NAME_STOP = re.compile(
    r"^\s*(?:lot|exp|ndc|rx|mfg|dist|manufactured|directions|warning|"
    r"storage|keep|shake|refills?|pharmacist|dispense|quantity|qty|"
    r"patient|dr\b|dr\.)\b",
    re.IGNORECASE,
)


class PathAOCR:
    """
    PaddleOCR wrapper for flat labels (boxes, blister packs, cream tubes).

    Usage
    -----
    ocr = PathAOCR()
    lines = ocr.run(cv2_image)
    # lines → [{"text": "...", "confidence": 0.98}, ...]
    """

    def __init__(self, lang: str = "en") -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed. "
                "Run: pip install paddleocr && pip install -r requirements-paddle.txt"
            ) from exc
        # use_textline_orientation replaces use_angle_cls in PaddleOCR 3.x
        self._ocr = PaddleOCR(use_textline_orientation=True, lang=lang)

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        img,  # np.ndarray (BGR, from cv2.imread)
        *,
        preprocess: bool = True,
        auto_page_rotate: bool = False,
        page_rotate_mode: Literal["heuristic", "ocr", "none"] = "heuristic",
        min_confidence: float = 0.0,
    ) -> list[dict]:
        """
        Run OCR on a cv2 BGR image.

        Returns
        -------
        list of {"text": str, "confidence": float}, ordered top-to-bottom.
        Lines with confidence < min_confidence are excluded.
        """
        if preprocess:
            img = _preprocess_image(img)

        if auto_page_rotate and page_rotate_mode != "none":
            img = _auto_rotate(img, mode=page_rotate_mode)

        # PaddleOCR 3.x: predict() returns a list of result objects;
        # each converts to a dict with rec_texts / rec_scores arrays.
        paddle_results = self._ocr.predict(img)

        lines: list[dict] = []
        for page_result in (paddle_results or []):
            d = dict(page_result)
            texts  = d.get("rec_texts", []) or []
            scores = d.get("rec_scores", []) or []
            for text, conf in zip(texts, scores):
                if float(conf) >= min_confidence and text.strip():
                    lines.append({"text": text.strip(), "confidence": float(conf)})

        return lines


# ── Path A entry point ────────────────────────────────────────────────────────

def run_path_a(image_path: str) -> OCRResult:
    """
    Path A — PaddleOCR for flat labels (boxes, blister packs, cream tubes).

    Returns an OCRResult.  If mean confidence across all detected lines is
    below 75 %, the 'reupload_required' flag is set in hallucination_flags
    and the caller (or UI) should ask the user to retake the photo.
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image file: {image_path!r}")

    ocr = PathAOCR()
    lines = ocr.run(img, preprocess=True)

    if not lines:
        return OCRResult(
            path_used="paddle_ocr",
            confidence=0.0,
            hallucination_flags=["no_text_detected", "reupload_required"],
        )

    mean_conf = sum(ln["confidence"] for ln in lines) / len(lines)
    all_text = "\n".join(ln["text"] for ln in lines)

    if mean_conf < _CONFIDENCE_THRESHOLD:
        logger.warning(
            "Path A confidence %.2f < %.2f — requesting reupload for %s",
            mean_conf, _CONFIDENCE_THRESHOLD, image_path,
        )
        return OCRResult(
            raw_text=all_text,
            path_used="paddle_ocr",
            confidence=mean_conf,
            hallucination_flags=["confidence_too_low", "reupload_required"],
        )

    drug_name = _extract_drug_name(lines)
    dosage    = _extract_dosage(lines)

    return OCRResult(
        drug_name=drug_name,
        dosage=dosage,
        raw_text=all_text,
        path_used="paddle_ocr",
        confidence=mean_conf,
    )


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _preprocess_image(img):
    """CLAHE contrast enhancement → adaptive threshold → return BGR."""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive threshold sharpens text against varied backgrounds
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )

    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _auto_rotate(img, *, mode: str = "heuristic"):
    """
    Rotate the whole image so text reads left-to-right.

    heuristic: choose 0 or 90 based on aspect ratio (landscape → 0, portrait
               with tall ratio → rotate 90).
    ocr:       run PaddleOCR twice (0° and 90°), pick the angle that gives
               more detected text — slower but more accurate.
    """
    import cv2
    import numpy as np

    h, w = img.shape[:2]

    if mode == "heuristic":
        if h > w * 1.3:          # noticeably taller than wide → rotate
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    if mode == "ocr":
        from paddleocr import PaddleOCR  # type: ignore
        _tmp = PaddleOCR(use_textline_orientation=False, lang="en")

        def _count(candidate):
            res = _tmp.predict(candidate) or []
            return sum(len(dict(r).get("rec_texts") or []) for r in res)

        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return rotated if _count(rotated) > _count(img) else img

    return img


# ── Field extraction helpers ──────────────────────────────────────────────────

def _extract_drug_name(lines: list[dict]) -> str:
    """
    Heuristic: drug name is the first high-confidence line near the top that
    is NOT a metadata keyword.  Lines containing ONLY a dosage unit are still
    accepted — the drug name and dosage often appear together (e.g.
    "AMOXICILLIN 500 MG").
    """
    for ln in lines:
        text = ln["text"]
        if (
            ln["confidence"] >= 0.80
            and len(text) >= 3
            and not _DRUG_NAME_STOP.match(text)
        ):
            return text
    return ""


def _extract_dosage(lines: list[dict]) -> str:
    """
    Return the dosage expression.  The drug name line often contains it
    (e.g. "AMOXICILLIN 500 MG") so we extract just the matched substring
    rather than the whole line when it would duplicate the drug name.
    """
    drug_name = _extract_drug_name(lines)
    for ln in lines:
        m = _DOSAGE_UNITS.search(ln["text"])
        if m:
            text = ln["text"].strip()
            # If this line IS the drug name, return only the dosage portion
            if text == drug_name:
                return m.group(0).strip()
            return text
    return ""


# ── Path B — xAI Vision ────────────────────────────────────────────────────────

def run_path_b(image_path: str, api_key: Optional[str] = None) -> OCRResult:
    """
    Path B — xAI vision for cylindrical bottles.

    Takes a photo of a curved bottle, sends it to the xAI vision model,
    parses the structured JSON response, then runs hallucination checks.

    Parameters
    ----------
    image_path : path to the bottle photo on disk
    api_key    : xAI API key — falls back to XAI_API_KEY in .env

    Returns
    -------
    OCRResult with path_used = "xai_vision"
    """
    key = require_xai_api_key(api_key)

    image_b64, mime_type = _encode_image(image_path)
    raw_response = vision_completion(
        image_b64,
        mime_type,
        PATH_B_VISION_PROMPT,
        api_key=key,
    )
    result = _parse_vision_response(raw_response)
    result = _check_hallucinations(result)

    return result


# ── Path B vision helpers ─────────────────────────────────────────────────────

def _encode_image(image_path: str) -> tuple[str, str]:
    """Read image from disk, return (base64_string, mime_type)."""
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime_map = {
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "png":  "image/png",
    "webp": "image/webp",
    "heic": "image/heic",
    "heif": "image/heif",
    }
    mime_type = mime_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    return image_b64, mime_type


def _parse_vision_response(raw: str) -> OCRResult:
    """
    Parse VLM JSON text response into an OCRResult.
    Strips markdown code fences if the model wrapped JSON in them.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().strip("`").strip()

    try:
        data = json.loads(cleaned)
        return OCRResult(
            drug_name=          data.get("drug_name", "").strip(),
            dosage=             data.get("dosage", "").strip(),
            active_ingredients= data.get("active_ingredients", "").strip(),
            warnings=           data.get("warnings", "").strip(),
            directions=         data.get("directions", "").strip(),
            expiry_date=        data.get("expiry_date", "").strip(),
            raw_text=           raw,
            path_used=          "xai_vision",
            confidence=         0.9,
        )
    except json.JSONDecodeError:
        logger.warning("Vision model returned non-JSON response: %s", raw[:200])
        return OCRResult(
            raw_text=            raw,
            path_used=           "xai_vision",
            confidence=          0.3,
            hallucination_flags= ["response_not_json"],
        )


# ── Hallucination checking ─────────────────────────────────────────────────────

def _check_hallucinations(result: OCRResult) -> OCRResult:
    """
    Sanity checks on VLM output to catch hallucinated fields.

    Checks:
      1. drug_name must not be empty
      2. no placeholder text like "N/A", "unknown", "not visible"
      3. dosage must contain a number if present
      4. expiry date must look like a real date if present
      5. no repeated gibberish (same word 4+ times in a row)
      6. confidence must be above 0.5

    Any failed check adds a flag to hallucination_flags.
    """
    flags = list(result.hallucination_flags)

    # 1. Drug name must be present
    if not result.drug_name.strip():
        flags.append("drug_name_empty")

    # 2. No placeholder text in key fields
    placeholder_pattern = re.compile(
        r"\bnot\s+available\b|\bn/?a\b|\bunknown\b|\bnot\s+visible\b"
        r"|\bnot\s+legible\b|\bcannot\s+read\b|\bunable\s+to\b",
        re.IGNORECASE
    )
    for field_name, value in [
        ("drug_name",          result.drug_name),
        ("dosage",             result.dosage),
        ("active_ingredients", result.active_ingredients),
    ]:
        if value and placeholder_pattern.search(value):
            flags.append(f"placeholder_text_in_{field_name}")

    # 3. Dosage should contain a number
    if result.dosage and not re.search(r'\d', result.dosage):
        flags.append("dosage_has_no_number")

    # 4. Expiry date should look like a date
    if result.expiry_date:
        date_pattern = re.compile(
            r'\d{1,4}[-/]\d{1,2}([-/]\d{1,4})?'
            r'|[A-Za-z]{3,9}\s+\d{4}'
            r'|\d{2}/\d{4}'
        )
        if not date_pattern.search(result.expiry_date):
            flags.append("expiry_date_invalid_format")

    # 5. No repeated gibberish
    for field_name, value in [
        ("drug_name",          result.drug_name),
        ("active_ingredients", result.active_ingredients),
    ]:
        words = value.lower().split()
        for i in range(len(words) - 3):
            if words[i] == words[i+1] == words[i+2] == words[i+3]:
                flags.append(f"repeated_gibberish_in_{field_name}")
                break

    # 6. Confidence threshold
    if result.confidence < 0.5:
        flags.append("confidence_too_low")

    # Apply flags and reduce confidence score
    result.hallucination_flags = flags
    if flags:
        result.confidence = max(0.1, result.confidence - 0.2 * len(flags))
        logger.warning(
            "Hallucination flags for '%s': %s", result.drug_name, flags
        )

    return result


# ── Router ─────────────────────────────────────────────────────────────────────

def run_ocr(image_path: str, packaging_type: str) -> OCRResult:
    """
    Main entry point called by the rest of the pipeline.

    Parameters
    ----------
    image_path     : path to the image file
    packaging_type : "flat"         → Path A (PaddleOCR)
                     "cylindrical"  → Path B (xAI Vision)
    """
    if packaging_type == "cylindrical":
        return run_path_b(image_path)
    elif packaging_type == "flat":
        return run_path_a(image_path)
    else:
        raise ValueError(
            f"Unknown packaging_type '{packaging_type}'. "
            "Expected 'flat' or 'cylindrical'."
        )
