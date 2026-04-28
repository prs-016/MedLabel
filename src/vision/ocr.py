"""
Path A: flat medicine labels → OpenCV preprocessing → PaddleOCR.

Preprocessing matches the MedLabel spec: CLAHE (local contrast) plus adaptive
thresholding to stabilize ink on uneven lighting before recognition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .orientation import apply_rotation, auto_orient_image, estimate_rotation_degrees, refine_rotation_with_ocr_score

# OpenCV 4.x Python uses ADAPTIVE_THRESH_GAUSSIAN_C; very old tutorials used ADAPTIVE_GAUSSIAN_C.
_CV_ADAPTIVE_GAUSSIAN = getattr(
    cv2, "ADAPTIVE_THRESH_GAUSSIAN_C", getattr(cv2, "ADAPTIVE_GAUSSIAN_C", 1)
)


@dataclass
class PathAOCRConfig:
    """Tunable OpenCV preprocessing for flat labels."""

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
    """
    Grayscale → CLAHE → adaptive threshold → BGR for PaddleOCR.

    ``image_bgr`` is uint8 OpenCV BGR. Returns uint8 BGR (single-channel
    threshold map replicated to 3 channels) so PaddleOCR receives a consistent
    3-channel tensor.
    """
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
    """
    Lazy PaddleOCR wrapper for Path A (English flat labels).

    ``use_angle_cls`` lets Paddle fix local 180° flips on text lines; combine
    with page-level rotation (see ``auto_page_rotate``) for 90° steps.
    """

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
                "Path A failed to import PaddleOCR/PaddlePaddle." + extra + f" Original error: {e!r}"
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
        """Raw PaddleOCR output (list structure) for debugging or orientation scoring."""
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
        """
        Full Path A pipeline on a single BGR image.

        ``page_rotate_mode``:
          - ``"heuristic"`` — fast projection-based 0/90/180/270 (default when auto_page_rotate=True)
          - ``"ocr"`` — try four rotations with Paddle and pick best aggregate score (slower, robust)
          - ``"none"`` — skip even if auto_page_rotate is True (same as auto_page_rotate=False)
        """
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
    """
    Convenience one-shot using a shared ``engine`` or a fresh ``PathAOCR()`` if none passed.

    Prefer reusing ``PathAOCR`` in a loop to avoid reloading weights.
    """
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
    """Expose heuristic 0/90/180/270 estimate for other modules (no Paddle import)."""
    return estimate_rotation_degrees(image_bgr)
