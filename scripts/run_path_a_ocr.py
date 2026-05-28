#!/usr/bin/env python3
"""
Run Path A (PaddleOCR) on flat medicine label images from the command line.

Usage examples
--------------
  # Single image, plain text
  PYTHONPATH=src python3 scripts/run_path_a_ocr.py -i data/raw/images/sample_ocr_test.png

  # Single image, structured JSON
  PYTHONPATH=src python3 scripts/run_path_a_ocr.py -i data/raw/images/sample_ocr_test.png --json

  # Whole folder, JSON saved to file
  PYTHONPATH=src python3 scripts/run_path_a_ocr.py --json \\
      -i /path/to/Flat_images/*.png > results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
import logging
from pathlib import Path


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src  = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _result_to_dict(result, image_path: str) -> dict:
    """Convert an OCRResult to a clean JSON-serialisable dict."""
    reupload = "reupload_required" in result.hallucination_flags

    if reupload:
        if result.confidence == 0.0:
            reason = "No text was detected in the image."
        else:
            reason = (
                f"Confidence too low ({result.confidence:.0%}). "
                "The label could not be read reliably."
            )
        return {
            "image":             image_path,
            "status":            "reupload_required",
            "confidence":        round(result.confidence, 4),
            "reupload_message":  (
                f"⚠️  Please re-upload a clearer photo. {reason} "
                "Tips: use good lighting, hold the camera steady, "
                "and make sure the full label is flat and in frame."
            ),
            "drug_name":         None,
            "dosage":            None,
            "active_ingredients": None,
            "warnings":          None,
            "directions":        None,
            "expiry_date":       None,
            "other_text":        None,
        }

    return {
        "image":              image_path,
        "status":             "ok",
        "confidence":         round(result.confidence, 4),
        "reupload_message":   None,
        "drug_name":          result.drug_name          or None,
        "dosage":             result.dosage             or None,
        "active_ingredients": result.active_ingredients or None,
        "warnings":           result.warnings           or None,
        "directions":         result.directions         or None,
        "expiry_date":        result.expiry_date        or None,
        "other_text":         result.raw_text           or None,
    }


def _print_plain(result, image_path: str) -> None:
    """Human-readable plain-text output for a single image."""
    reupload = "reupload_required" in result.hallucination_flags
    sep = "─" * 60

    print(sep)
    print(f"Image      : {image_path}")
    print(f"Confidence : {result.confidence:.1%}")

    if reupload:
        if result.confidence == 0.0:
            reason = "no text detected"
        else:
            reason = f"confidence {result.confidence:.1%} < 75%"
        print(f"Status     : ❌  Re-upload required ({reason})")
        print()
        print("  ⚠️  Please re-upload a clearer photo of the label.")
        print("      Tips: good lighting · camera steady · label flat and fully visible")
    else:
        print(f"Status     : ✅  OK")
        print()
        if result.drug_name:
            print(f"  Drug name          : {result.drug_name}")
        if result.dosage:
            print(f"  Dosage             : {result.dosage}")
        if result.active_ingredients:
            print(f"  Active ingredients : {result.active_ingredients}")
        if result.warnings:
            print(f"  Warnings           : {result.warnings}")
        if result.directions:
            print(f"  Directions         : {result.directions}")
        if result.expiry_date:
            print(f"  Expiry date        : {result.expiry_date}")
        if result.raw_text:
            other_lines = [
                ln for ln in result.raw_text.splitlines()
                if ln not in (
                    result.drug_name, result.dosage,
                    result.active_ingredients, result.warnings,
                    result.directions, result.expiry_date,
                )
            ]
            if other_lines:
                print(f"  Other text         : {' | '.join(other_lines)}")
    print()


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    _setup_path()
    import cv2
    from vision.ocr import run_path_a

    p = argparse.ArgumentParser(
        description="Path A OCR — read flat medicine labels and extract structured sections."
    )
    p.add_argument(
        "-i", "--input", required=True, type=Path, nargs="+",
        help="One or more image paths (supports glob: Flat_images/*.png)",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Print structured JSON array instead of plain text",
    )
    args = p.parse_args()

    all_out: list[dict] = []

    for path in args.input:
        if not path.is_file():
            print(f"skip (not a file): {path}", file=sys.stderr)
            continue

        result = run_path_a(str(path))

        if args.json:
            all_out.append(_result_to_dict(result, str(path)))
        else:
            _print_plain(result, str(path))

    if args.json:
        print(json.dumps(all_out, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
