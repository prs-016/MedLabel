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

import requests

logger = logging.getLogger(__name__)

_WORKSPACE   = "akshays-workspace-wnzzr"
_WORKFLOW    = "router"
_API_BASE    = "https://serverless.roboflow.com"
_TIMEOUT_SEC = 20


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

    def detect_geometry(
        self,
        image_source: str,  # path to image file
    ) -> tuple[str, list[float], float]:
        """
        Call Roboflow YOLO workflow and return geometry classification.

        Returns
        -------
        (packaging_type, bbox, confidence)
            packaging_type : "flat" | "cylindrical"
            bbox           : placeholder [0, 0, 100, 100]
            confidence     : 0-1
        """
        if not self._trained:
            return "flat", [0.0, 0.0, 100.0, 100.0], 0.5

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
            top            = model_output["top"]
            conf           = float(model_output["confidence"])
            packaging_type = "cylindrical" if "cyl" in top.lower() else "flat"
            logger.info("Roboflow detected: %s (conf=%.2f)", packaging_type, conf)
            return packaging_type, [0.0, 0.0, 100.0, 100.0], conf

        except Exception as exc:
            logger.error("Roboflow inference error: %s — defaulting to flat", exc)
            return "flat", [0.0, 0.0, 100.0, 100.0], 0.5


if __name__ == "__main__":
    router = YOLORouter()
    print(f"YOLORouter initialised — Roboflow connected: {router._trained}")
    if router._trained:
        import sys
        if len(sys.argv) > 1:
            result = router.detect_geometry(sys.argv[1])
            print(f"Detection result: {result}")
