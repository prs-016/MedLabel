"""
src/vision/detector.py
======================
Routes medicine label images via the Roboflow YOLO workflow (primary).
Uses the Roboflow Workflows REST API directly (no inference-sdk needed),
compatible with Python 3.14+.

Falls back gracefully when ROBOFLOW_API_KEY is absent or the API call
fails — callers then use xAI Vision for auto-detection.

Roboflow workspace : akshays-workspace-wnzzr
Roboflow workflow  : router

Response structure:
    data["outputs"][0]["model_output"]["top"]         → "Cylindrical" | "flat_label"
    data["outputs"][0]["model_output"]["confidence"]  → float 0-1
"""

from __future__ import annotations

import base64
import io
import logging
import os
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

_WORKSPACE   = "akshays-workspace-wnzzr"
_WORKFLOW    = "router"
_API_BASE    = "https://serverless.roboflow.com"
_TIMEOUT_SEC = 20

# If YOLO says cylindrical below this confidence, try flat OCR as well.
_LOW_CONF_CYLINDRICAL_OVERRIDE = 0.85

# Portrait + flat → cylindrical only when YOLO is not very confident.
_ASPECT_PORTRAIT_OVERRIDE_MAX_CONF = 0.90

# Width/height above this → treat as box-like (landscape or square front-on).
_FLAT_ASPECT_MIN_RATIO = 0.85


@dataclass(frozen=True)
class GeometryDetection:
    packaging_type: str  # "flat" | "cylindrical"
    confidence: float
    raw_label: str
    bbox: list[float]


def _parse_roboflow_top(top: str) -> str:
    """Map Roboflow class name to flat vs cylindrical."""
    t = (top or "").strip().lower().replace(" ", "_").replace("-", "_")
    if any(k in t for k in ("cylindrical", "cylinder", "bottle", "curved", "round")):
        return "cylindrical"
    if any(k in t for k in ("flat", "box", "blister", "carton", "sachet")):
        return "flat"
    if "cyl" in t and "flat" not in t:
        return "cylindrical"
    return "flat"


def _aspect_ratio_bias(image_source: str) -> str | None:
    """
    Box labels are usually shot landscape or square; bottles are usually tall.
    Square front-on box photos (ratio ~1.0) were missing the old 1.05 landscape cut.
    """
    try:
        from PIL import Image

        w, h = Image.open(image_source).size
        if h <= 0:
            return None
        ratio = w / h
        if ratio >= _FLAT_ASPECT_MIN_RATIO:
            return "flat"
        if ratio <= 0.72:
            return "cylindrical"
    except Exception:
        pass
    return None


class YOLORouter:

    def __init__(self):
        self._trained  = False
        self._api_key  = ""
        self._endpoint = ""
        self._load()

    def _load(self) -> None:
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            logger.warning(
                "ROBOFLOW_API_KEY not set — auto-detect will use xAI Vision fallback"
            )
            return
        self._api_key  = api_key
        self._endpoint = f"{_API_BASE}/infer/workflows/{_WORKSPACE}/{_WORKFLOW}"
        self._trained  = True
        logger.info(
            "YOLORouter ready — Roboflow workspace=%s workflow=%s",
            _WORKSPACE, _WORKFLOW,
        )

    def detect_geometry(self, image_source: str) -> GeometryDetection:
        """
        Call Roboflow YOLO workflow and return geometry classification.
        """
        if not self._trained:
            return GeometryDetection("flat", 0.5, "", [0.0, 0.0, 100.0, 100.0])

        try:
            # Roboflow requires JPEG — normalise before encoding
            from PIL import Image
            img = Image.open(image_source).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            payload = {
                "api_key": self._api_key,
                "inputs": {
                    "image": {
                        "type":  "base64",
                        "value": image_b64,
                    }
                },
            }
            resp = requests.post(
                self._endpoint,
                json=payload,
                timeout=_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            data = resp.json()

            model_output   = data["outputs"][0]["model_output"]
            top            = str(model_output.get("top", ""))
            conf           = float(model_output.get("confidence", 0.0))
            packaging_type = _parse_roboflow_top(top)

            aspect_hint = _aspect_ratio_bias(image_source)
            if aspect_hint == "flat" and packaging_type == "cylindrical":
                logger.info(
                    "Roboflow said %s (%.0f%%) but photo is box-shaped "
                    "(w/h >= %.2f) — using flat",
                    top, conf * 100, _FLAT_ASPECT_MIN_RATIO,
                )
                packaging_type = "flat"
            elif (
                aspect_hint == "cylindrical"
                and packaging_type == "flat"
                and conf < _ASPECT_PORTRAIT_OVERRIDE_MAX_CONF
            ):
                logger.info(
                    "Roboflow said %s (%.0f%%) but image is portrait — using cylindrical",
                    top, conf * 100,
                )
                packaging_type = "cylindrical"

            logger.info(
                "Roboflow: raw=%r → %s (conf=%.2f)", top, packaging_type, conf
            )
            return GeometryDetection(
                packaging_type, conf, top, [0.0, 0.0, 100.0, 100.0]
            )

        except Exception as exc:
            logger.error("Roboflow inference error: %s — defaulting to flat", exc)
            return GeometryDetection("flat", 0.5, "", [0.0, 0.0, 100.0, 100.0])


if __name__ == "__main__":
    router = YOLORouter()
    print(f"YOLORouter initialised — Roboflow connected: {router._trained}")
    if router._trained:
        import sys
        if len(sys.argv) > 1:
            result = router.detect_geometry(sys.argv[1])
            print(f"Detection result: {result}")
