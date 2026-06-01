"""
Load images for OCR and UI preview, including iPhone HEIC/HEIF photos.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

_HEIF_REGISTERED = False
_HEIC_SUFFIXES = {".heic", ".heif"}
_HEIC_FTYP_BRANDS = frozenset(
    {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heis", b"heim"}
)


def register_heif_opener() -> None:
    """Register HEIC/HEIF with Pillow (idempotent)."""
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _HEIF_REGISTERED = True
    except ImportError as exc:
        raise ImportError(
            "HEIC/HEIF images require pillow-heif. "
            "Install with: pip install pillow-heif"
        ) from exc


def is_heic_path(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in _HEIC_SUFFIXES


def _looks_like_heic(data: bytes) -> bool:
    """Detect iPhone HEIC/HEIF from ISO BMFF magic (ftyp brand)."""
    if len(data) < 12:
        return False
    if data[4:8] != b"ftyp":
        return False
    return data[8:12] in _HEIC_FTYP_BRANDS


def needs_heif(filename: str | None = None, data: bytes | None = None) -> bool:
    if filename and Path(filename).suffix.lower() in _HEIC_SUFFIXES:
        return True
    if data and _looks_like_heic(data):
        return True
    return False


def open_image(
    path: str | os.PathLike | None = None,
    data: bytes | None = None,
    *,
    filename: str | None = None,
):
    """Open a PIL Image; registers HEIF opener when needed."""
    name = filename or (str(path) if path is not None else None)
    if needs_heif(name, data):
        register_heif_opener()

    def _open() -> Image.Image:
        if data is not None:
            return Image.open(io.BytesIO(data))
        if path is not None:
            return Image.open(path)
        raise ValueError("open_image requires path or data")

    try:
        return _open()
    except UnidentifiedImageError:
        if not _HEIF_REGISTERED:
            register_heif_opener()
            return _open()
        raise


def load_bgr(path: str | os.PathLike):
    """Load image as OpenCV BGR ndarray (HEIC → RGB → BGR)."""
    import cv2
    import numpy as np

    path = str(path)
    if is_heic_path(path) or needs_heif(path):
        register_heif_opener()
        rgb = np.array(open_image(path).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    img = cv2.imread(path)
    if img is not None:
        return img

    rgb = np.array(open_image(path).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def encode_for_vision_api(
    path: str | os.PathLike | None = None,
    data: bytes | None = None,
    *,
    filename: str | None = None,
) -> tuple[str, str]:
    """
    Base64 + MIME for xAI / Gemini vision APIs.

    HEIC/HEIF are decoded and re-encoded as JPEG — most vision APIs do not
    accept image/heic in data URLs.
    """
    import base64

    name = filename or (str(path) if path is not None else None)
    raw: bytes | None = data
    if path is not None and raw is None:
        with open(path, "rb") as f:
            raw = f.read()

    if raw is not None and (needs_heif(name, raw) or _looks_like_heic(raw)):
        register_heif_opener()
        img = open_image(data=raw, filename=name).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        raw = buf.getvalue()
        return base64.b64encode(raw).decode("utf-8"), "image/jpeg"

    if path is not None and is_heic_path(path):
        register_heif_opener()
        img = open_image(path=path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"

    if raw is not None:
        ext = Path(name or ".jpg").suffix.lower().lstrip(".")
        mime_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }
        return base64.b64encode(raw).decode("utf-8"), mime_map.get(ext, "image/jpeg")

    raise ValueError("encode_for_vision_api requires path or data")


def materialize_upload(
    data: bytes,
    filename: str,
    *,
    for_opencv: bool = False,
    for_vision: bool = False,
) -> str:
    """
    Write upload bytes to a temp file OCR can read.

    HEIC/HEIF → PNG for Paddle (for_opencv), JPEG for xAI vision (for_vision).
    """
    heic = needs_heif(filename, data)
    if heic and (for_opencv or for_vision):
        img = open_image(data=data, filename=filename).convert("RGB")
        if for_opencv:
            suffix, fmt = ".png", "PNG"
        else:
            suffix, fmt = ".jpg", "JPEG"
        fd, out_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        img.save(out_path, fmt, quality=92)
        return out_path

    suffix = (Path(filename).suffix or ".png").lower()
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    fd, out_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path
