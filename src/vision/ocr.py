import os
import re
import base64
import logging
import json
from dataclasses import dataclass, field
from typing import Optional
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


# ── Path A stub ────────────────────────────────────────────────────────────────

def run_path_a(image_path: str) -> OCRResult:
    """
    Path A — PaddleOCR for flat labels (boxes, blister packs, cream tubes).
    To be implemented with PaddleOCR + OpenCV preprocessing.
    """
    raise NotImplementedError("Path A (PaddleOCR) not yet implemented.")


# ── Path B — Gemini Vision ─────────────────────────────────────────────────────

def run_path_b(image_path: str, api_key: Optional[str] = None) -> OCRResult:
    """
    Path B — Gemini Vision for cylindrical bottles.

    Takes a photo of a curved bottle, sends it to Gemini 2.0 Flash,
    parses the structured JSON response, then runs hallucination checks.

    Parameters
    ----------
    image_path : path to the bottle photo on disk
    api_key    : Gemini API key — falls back to GEMINI_API_KEY in .env

    Returns
    -------
    OCRResult with path_used = "gemini_vision"
    """
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError(
            "No Gemini API key found. "
            "Add GEMINI_API_KEY=your_key to your .env file."
        )

    image_b64, mime_type = _encode_image(image_path)
    raw_response         = _call_gemini(image_b64, mime_type, key)
    result               = _parse_gemini_response(raw_response)
    result               = _check_hallucinations(result)

    return result


# ── Gemini helpers ─────────────────────────────────────────────────────────────

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
    packaging_type : "flat"         → Path A (PaddleOCR)
                     "cylindrical"  → Path B (Gemini Vision)
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