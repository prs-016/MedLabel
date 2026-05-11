#!/usr/bin/env python3
"""
CLI: estimate or apply 0/90/180/270 page rotation for flat-label OCR.

Examples:
  python scripts/auto_orient.py -i data/raw/images/label.jpg -o data/raw/images/label_upright.jpg
  python scripts/auto_orient.py -i label.jpg --print-only
  python scripts/auto_orient.py -i label.jpg -o out.jpg --refine-ocr   # slower, uses Paddle scores
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _setup_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def main() -> int:
    _setup_path()
    import cv2
    from vision.orientation import apply_rotation, auto_orient_image

    p = argparse.ArgumentParser(description="Auto-orient images for Path A OCR (90° steps).")
    p.add_argument("-i", "--input", required=True, type=Path, help="Input image path")
    p.add_argument("-o", "--output", type=Path, default=None, help="Write corrected BGR image here")
    p.add_argument("--print-only", action="store_true", help="Only print chosen angle; do not write")
    p.add_argument(
        "--refine-ocr",
        action="store_true",
        help="Score 0/90/180/270 with PaddleOCR (more accurate, slower, needs paddle installed)",
    )
    args = p.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    img = cv2.imread(str(args.input))
    if img is None:
        print(f"Failed to read image: {args.input}", file=sys.stderr)
        return 1

    if args.refine_ocr:
        from vision.orientation import refine_rotation_with_ocr_score
        from vision.ocr import PathAOCR

        ocr = PathAOCR()
        angle = refine_rotation_with_ocr_score(img, lambda im: ocr.ocr_raw(im))
        corrected = apply_rotation(img, angle)
        method = "ocr-refined"
    else:
        corrected, angle = auto_orient_image(img)
        method = "heuristic"

    print(f"method={method} angle_deg={angle}")

    if args.print_only:
        return 0
    if args.output is None:
        print("No --output given; use --print-only or set -o", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), corrected)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
