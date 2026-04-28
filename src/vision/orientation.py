"""
Heuristic page-level orientation (0, 90, 180, 270) for flat labels.

Uses horizontal ink projection statistics so it runs without extra models.
Optional OCR scoring refines the choice when PaddleOCR is available.
"""

from __future__ import annotations

import cv2
import numpy as np


def _rotate_bound(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image about center; expand canvas so corners are not clipped."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    m[0, 2] += nw / 2.0 - center[0]
    m[1, 2] += nh / 2.0 - center[1]
    return cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _projection_score(gray_binary: np.ndarray) -> float:
    """
    Score how much ink aligns with horizontal rows (typical for upright Latin text).
    Higher is better for 0° vs 90°/270° mis-rotation on many label layouts.
    """
    ink = (gray_binary < 200).astype(np.float32)
    row_sum = ink.sum(axis=1)
    if row_sum.size < 2:
        return 0.0
    # Penalize flat projections; text lines create stronger row-to-row variation
    return float(np.std(row_sum) + 0.25 * np.std(np.diff(row_sum)))


def _prep_for_orientation(gray: np.ndarray, max_side: int = 960) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _adapt = getattr(cv2, "ADAPTIVE_THRESH_GAUSSIAN_C", getattr(cv2, "ADAPTIVE_GAUSSIAN_C", 1))
    return cv2.adaptiveThreshold(
        gray,
        255,
        _adapt,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )


def estimate_rotation_degrees(image_bgr: np.ndarray) -> int:
    """
    Pick one of {0, 90, 180, 270} that best matches upright horizontal text layout.

    Heuristic only — fails on square symmetric patterns or vertical-only layouts.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    best_angle = 0
    best_score = -1.0
    for angle in (0, 90, 180, 270):
        if angle == 0:
            rotated = gray
        else:
            rotated = cv2.cvtColor(_rotate_bound(image_bgr, float(angle)), cv2.COLOR_BGR2GRAY)
        bin_img = _prep_for_orientation(rotated)
        s = _projection_score(bin_img)
        if s > best_score:
            best_score = s
            best_angle = angle
    return int(best_angle)


def apply_rotation(image_bgr: np.ndarray, angle_deg: int) -> np.ndarray:
    """Apply one of {0, 90, 180, 270}."""
    angle_deg = int(angle_deg) % 360
    if angle_deg == 0:
        return image_bgr.copy()
    if angle_deg in (90, 180, 270):
        return _rotate_bound(image_bgr, float(angle_deg))
    raise ValueError("angle_deg must be a multiple of 90")


def auto_orient_image(
    image_bgr: np.ndarray,
    *,
    angle_override: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Return (corrected_bgr, applied_angle). ``applied_angle`` is in {0,90,180,270}.
    """
    angle = int(angle_override) if angle_override is not None else estimate_rotation_degrees(image_bgr)
    angle = angle % 360
    if angle not in (0, 90, 180, 270):
        raise ValueError("angle_override must be a multiple of 90")
    return apply_rotation(image_bgr, angle), angle


def refine_rotation_with_ocr_score(
    image_bgr: np.ndarray,
    ocr_predict,
    *,
    angles: tuple[int, ...] = (0, 90, 180, 270),
) -> int:
    """
    Pick rotation by running OCR once per candidate. ``ocr_predict`` is a callable
    that accepts BGR uint8 and returns the raw PaddleOCR-style result list.
    """
    best_a = 0
    best_score = -1.0
    for a in angles:
        img = apply_rotation(image_bgr, a) if a else image_bgr.copy()
        raw = ocr_predict(img)
        score = _paddle_result_score(raw)
        if score > best_score:
            best_score = score
            best_a = a
    return int(best_a)


def _paddle_result_score(raw) -> float:
    if not raw or raw[0] is None:
        return 0.0
    lines = raw[0]
    total = 0.0
    for item in lines:
        if not item or len(item) < 2:
            continue
        text, conf = item[1]
        if text:
            total += float(conf) * max(1, len(text.strip()))
    return total
