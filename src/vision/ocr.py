import os
import re
import base64
import logging
import json
from dataclasses import dataclass, field
from typing import Optional, Literal
import requests

from dotenv import load_dotenv
load_dotenv()

load_dotenv()  # this actually reads the .env file
key = os.getenv("GEMINI_API_KEY")

logger = logging.getLogger(__name__)


# TO:
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


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class OCRResult:
    """
    Structured output from either OCR path.
    Path A (PaddleOCR) and Path B (Gemini Vision) both return this
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
            import paddleocr as _paddleocr_mod
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed. "
                "Run: pip install paddleocr && pip install -r requirements-paddle.txt"
            ) from exc
        major = int(_paddleocr_mod.__version__.split(".")[0])
        if major >= 3:
            self._ocr = PaddleOCR(use_textline_orientation=True, lang=lang)
        else:
            # PaddleOCR 2.x (what most installs have)
            self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        self._paddle_major = major

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

        lines: list[dict] = []
        if hasattr(self._ocr, "predict"):
            # PaddleOCR 3.x
            paddle_results = self._ocr.predict(img)
            for page_result in (paddle_results or []):
                d = dict(page_result)
                texts  = d.get("rec_texts", []) or []
                scores = d.get("rec_scores", []) or []
                for text, conf in zip(texts, scores):
                    if float(conf) >= min_confidence and text.strip():
                        lines.append({"text": text.strip(), "confidence": float(conf)})
        else:
            # PaddleOCR 2.x — .ocr() returns [[[box, (text, conf)], ...]]
            raw = self._ocr.ocr(img, cls=True)
            for page in (raw or []):
                if not page:
                    continue
                for item in page:
                    if not item or len(item) < 2:
                        continue
                    text, conf = item[1]
                    if float(conf) >= min_confidence and str(text).strip():
                        lines.append({"text": str(text).strip(), "confidence": float(conf)})

        return lines


# ── Path A entry point ────────────────────────────────────────────────────────

def run_path_a(image_path: str) -> OCRResult:
    """
    Path A — PaddleOCR for flat labels (boxes, blister packs, cream tubes).

    Confidence gate
    ---------------
    Mean confidence across all detected lines must be ≥ 75 %.
    If it falls below that threshold the result has no parsed fields and
    hallucination_flags includes "reupload_required" so callers / the UI
    can prompt the user to retake the photo.
    """
    from vision.image_io import load_bgr

    img = load_bgr(image_path)

    ocr   = PathAOCR()
    lines = ocr.run(img, preprocess=True)

    # ── No text at all ────────────────────────────────────────────────────────
    if not lines:
        return OCRResult(
            path_used="paddle_ocr",
            confidence=0.0,
            hallucination_flags=["no_text_detected", "reupload_required"],
        )

    mean_conf = sum(ln["confidence"] for ln in lines) / len(lines)
    all_text  = "\n".join(ln["text"] for ln in lines)

    # ── Below confidence threshold → ask user to re-upload ───────────────────
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

    # ── Extract all sections ──────────────────────────────────────────────────
    sections = _extract_sections(lines)

    return OCRResult(
        drug_name=          sections["drug_name"],
        dosage=             sections["dosage"],
        active_ingredients= sections["active_ingredients"],
        warnings=           sections["warnings"],
        directions=         sections["directions"],
        expiry_date=        sections["expiry_date"],
        raw_text=           all_text,
        path_used=          "paddle_ocr",
        confidence=         mean_conf,
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
        _tmp = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        def _count(candidate):
            if hasattr(_tmp, "predict"):
                res = _tmp.predict(candidate) or []
                return sum(len(dict(r).get("rec_texts") or []) for r in res)
            raw = _tmp.ocr(candidate, cls=True) or []
            return sum(len(page) for page in raw if page)

        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return rotated if _count(rotated) > _count(img) else img

    return img


# ── Section keyword anchors ───────────────────────────────────────────────────

# Keywords that open a labelled section in a medicine label
_SEC_ACTIVE = re.compile(
    r"\bactive\s+ingredient|\bingredients?\b|\bcontains\b", re.IGNORECASE
)
_SEC_WARNINGS = re.compile(
    r"\bwarnings?\b|\bcautions?\b|\bdo\s+not\b|\bstop\s+use\b|\bask\s+a\s+doctor\b",
    re.IGNORECASE,
)
_SEC_DIRECTIONS = re.compile(
    r"\bdirections?\b|\bhow\s+to\s+use\b|\bdosage\s+and\s+use\b",
    re.IGNORECASE,
)
_SEC_EXPIRY = re.compile(
    r"\bexp(?:iry|iration|\.?)?\s*:?\s*\d|\buse\s+by\b|\bbest\s+by\b|\buse\s+before\b",
    re.IGNORECASE,
)
_DATE_PATTERN = re.compile(
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\b"
    r"|\b\d{2}/\d{4}\b",
    re.IGNORECASE,
)

# Lines that should never be treated as the drug name
_DRUG_NAME_STOP = re.compile(
    r"^\s*(?:lot|exp|ndc|rx|mfg|dist|manufactured|directions|warning|"
    r"storage|keep|shake|refills?|pharmacist|dispense|quantity|qty|"
    r"patient|dr\b|dr\.)\b",
    re.IGNORECASE,
)


# ── Field extraction helpers ──────────────────────────────────────────────────

def _extract_drug_name(lines: list[dict]) -> str:
    """Known-brand scan across all lines, then first clean line as fallback."""
    from vision.brands import extract_product_name_from_ocr_lines

    name = extract_product_name_from_ocr_lines(lines)
    if name:
        return name

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
    Return the dosage string.  When the drug-name line already contains the
    dosage (e.g. "AMOXICILLIN 500 MG") only the matched token is returned so
    it does not duplicate the drug name field.
    """
    drug_name = _extract_drug_name(lines)
    for ln in lines:
        m = _DOSAGE_UNITS.search(ln["text"])
        if m:
            text = ln["text"].strip()
            return m.group(0).strip() if text == drug_name else text
    return ""


def _extract_sections(lines: list[dict]) -> dict:
    """
    Scan the detected lines top-to-bottom and bucket them into labelled
    sections using keyword anchors.

    Returns a dict with keys:
        drug_name, dosage, active_ingredients, warnings, directions,
        expiry_date, other_text
    Every value is a string (multi-line values joined with "; ").
    """
    drug_name = _extract_drug_name(lines)
    dosage    = _extract_dosage(lines)

    active_parts:     list[str] = []
    warning_parts:    list[str] = []
    direction_parts:  list[str] = []
    expiry_parts:     list[str] = []
    other_parts:      list[str] = []

    # Simple one-pass state machine: track which section we are currently in
    current_section: str | None = None

    for ln in lines:
        text = ln["text"].strip()
        if not text:
            continue

        # Skip lines already captured as drug_name or dosage
        if text == drug_name or text == dosage:
            continue

        # Check if this line opens a new section
        if _SEC_ACTIVE.search(text):
            current_section = "active"
        elif _SEC_WARNINGS.search(text):
            current_section = "warnings"
        elif _SEC_DIRECTIONS.search(text):
            current_section = "directions"
        elif _SEC_EXPIRY.search(text) or _DATE_PATTERN.search(text):
            current_section = "expiry"
        else:
            # Append to whichever section is currently open, or to other_text
            if current_section == "active":
                active_parts.append(text)
            elif current_section == "warnings":
                warning_parts.append(text)
            elif current_section == "directions":
                direction_parts.append(text)
            elif current_section == "expiry":
                expiry_parts.append(text)
            else:
                other_parts.append(text)
            continue

        # The anchor line itself goes into its own section too
        if current_section == "active":
            active_parts.append(text)
        elif current_section == "warnings":
            warning_parts.append(text)
        elif current_section == "directions":
            direction_parts.append(text)
        elif current_section == "expiry":
            expiry_parts.append(text)

    return {
        "drug_name":          drug_name,
        "dosage":             dosage,
        "active_ingredients": "; ".join(active_parts),
        "warnings":           "; ".join(warning_parts),
        "directions":         "; ".join(direction_parts),
        "expiry_date":        "; ".join(expiry_parts),
        "other_text":         "; ".join(other_parts),
    }


# ── Path B — xAI Grok Vision ───────────────────────────────────────────────────

XAI_VISION_MODEL = os.getenv("XAI_VISION_MODEL", "grok-4")
XAI_BASE_URL     = "https://api.x.ai/v1"

XAI_VISION_PROMPT = GEMINI_PROMPT  # same structured JSON extraction prompt


def run_path_b(image_path: str, api_key: Optional[str] = None) -> OCRResult:
    """
    Path B — xAI Grok Vision for cylindrical bottles (curved labels).

    Parameters
    ----------
    image_path : path to the bottle photo on disk
    api_key    : xAI API key — falls back to XAI_API_KEY in .env

    Returns
    -------
    OCRResult with path_used = "xai_vision"
    """
    key = api_key or os.getenv("XAI_API_KEY", "")
    if not key:
        raise ValueError(
            "No xAI API key found. "
            "Add XAI_API_KEY=your_key to your .env file."
        )

    image_b64, mime_type = _encode_image(image_path)
    raw_response         = _call_xai_vision(image_b64, mime_type, key)
    result               = _parse_vision_response(raw_response, path_used="xai_vision")
    result               = _check_hallucinations(result)

    return result


def _call_xai_vision(image_b64: str, mime_type: str, api_key: str) -> str:
    """Send image + prompt to xAI Grok vision, return raw text response."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=XAI_BASE_URL)
    resp = client.chat.completions.create(
        model=XAI_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                        },
                    },
                    {"type": "text", "text": XAI_VISION_PROMPT},
                ],
            }
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""


def _parse_vision_response(raw: str, *, path_used: str) -> OCRResult:
    """Parse a vision model's JSON response into an OCRResult."""
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
            path_used=          path_used,
            confidence=         0.9,
        )
    except json.JSONDecodeError:
        logger.warning("Vision model returned non-JSON response: %s", raw[:200])
        return OCRResult(
            raw_text=            raw,
            path_used=           path_used,
            confidence=          0.3,
            hallucination_flags= ["response_not_json"],
        )


# ── Legacy Gemini Path B (kept for reference, not used by run_ocr) ───────────

def _encode_image(image_path: str) -> tuple[str, str]:
    """Read image from disk; HEIC/HEIF → JPEG for vision API compatibility."""
    from vision.image_io import encode_for_vision_api

    return encode_for_vision_api(path=image_path)


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
                    {
                        "text": GEMINI_PROMPT
                    }
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
    """
    Parse Gemini's text response into an OCRResult.
    Strips markdown code fences if Gemini wrapped the JSON in them.
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
            path_used=          "gemini_vision",
            confidence=         0.9,
        )
    except json.JSONDecodeError:
        logger.warning("Gemini returned non-JSON response: %s", raw[:200])
        return OCRResult(
            raw_text=            raw,
            path_used=           "gemini_vision",
            confidence=          0.3,
            hallucination_flags= ["response_not_json"],
        )


# ── Hallucination checking ─────────────────────────────────────────────────────

def _check_hallucinations(result: OCRResult) -> OCRResult:
    """
    Sanity checks on what Gemini returned to catch hallucinated fields.

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
    packaging_type : "flat"         → Path A (PaddleOCR) with xAI Vision fallback
                     "cylindrical"  → Path B (xAI Grok Vision)
                     "auto"         → YOLO detect then route; xAI Vision fallback
    """
    if packaging_type == "cylindrical":
        return run_path_b(image_path)

    elif packaging_type == "flat":
        try:
            return run_path_a(image_path)
        except ImportError:
            # PaddleOCR not installed (Cloud deployment) — use xAI Vision
            result = run_path_b(image_path)
            result.path_used = "xai_vision (PaddleOCR unavailable — Cloud fallback)"
            return result

    elif packaging_type == "auto":
        try:
            from vision.detector import YOLORouter
            router = YOLORouter()
            detected, _bbox, _conf = router.detect_geometry(image_path)
            return run_ocr(image_path, detected)
        except ImportError:
            # ultralytics not installed (Cloud deployment) — use xAI Vision
            result = run_path_b(image_path)
            result.path_used = "xai_vision (YOLO unavailable — Cloud fallback)"
            return result

    else:
        raise ValueError(
            f"Unknown packaging_type '{packaging_type}'. "
            "Expected 'flat', 'cylindrical', or 'auto'."
        )
