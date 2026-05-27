#!/usr/bin/env python3
"""
Run Path A (flat) and Path B (cylindrical) OCR and print ``OCRResult.to_public_dict()``
as JSON so you can confirm the same schema for both paths.

Examples:
  export PYTHONPATH=src
  # Flat only (Paddle; no xAI structuring if you pass --no-structure-flat)
  python scripts/compare_ocr_paths.py --flat data/raw/images/sample_ocr_test.png

  # Both paths on the same file (Path B needs XAI_API_KEY / xai_api_key)
  python scripts/compare_ocr_paths.py --flat data/raw/images/sample_ocr_test.png \\
      --cylindrical data/raw/images/sample_ocr_test.png

  # Compare only JSON key sets (no values)
  python scripts/compare_ocr_paths.py --flat sample.png --cylindrical bottle.jpg --keys-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _setup_path()
    from vision.ocr import LABEL_SCHEMA_KEYS, run_ocr

    p = argparse.ArgumentParser(description="Compare flat vs cylindrical OCR JSON shape.")
    p.add_argument("--flat", type=Path, metavar="PATH", help="Image for Path A (flat / Paddle)")
    p.add_argument(
        "--cylindrical",
        type=Path,
        metavar="PATH",
        help="Image for Path B (xAI vision); needs XAI_API_KEY",
    )
    p.add_argument(
        "--no-structure-flat",
        action="store_true",
        help="Skip xAI-on-text for flat path (Paddle raw_text only; still same JSON keys)",
    )
    p.add_argument(
        "--keys-only",
        action="store_true",
        help="Print only whether top-level JSON keys match between paths",
    )
    args = p.parse_args()

    if not args.flat and not args.cylindrical:
        p.error("Pass at least one of --flat or --cylindrical")

    results: dict[str, dict] = {}

    if args.flat:
        if not args.flat.is_file():
            print(f"error: --flat file not found: {args.flat}", file=sys.stderr)
            return 1
        r = run_ocr(
            str(args.flat),
            "flat",
            structure_flat=not args.no_structure_flat,
        )
        results["flat"] = r.to_public_dict()

    if args.cylindrical:
        if not args.cylindrical.is_file():
            print(f"error: --cylindrical file not found: {args.cylindrical}", file=sys.stderr)
            return 1
        try:
            r = run_ocr(str(args.cylindrical), "cylindrical")
        except ValueError as e:
            print(f"Path B failed: {e}", file=sys.stderr)
            return 1
        results["cylindrical"] = r.to_public_dict()

    if args.keys_only and len(results) == 2:
        k1 = set(results["flat"])
        k2 = set(results["cylindrical"])
        same = k1 == k2
        print("keys_flat:", sorted(k1))
        print("keys_cylindrical:", sorted(k2))
        print("same_key_set:", same)
        if not same:
            print("only_flat:", sorted(k1 - k2))
            print("only_cylindrical:", sorted(k2 - k1))
        return 0 if same else 2

    for name, payload in results.items():
        print(f"========== {name} ==========")
        print(json.dumps(payload, indent=2))
        print()

    if len(results) == 2:
        k1 = set(results["flat"])
        k2 = set(results["cylindrical"])
        if k1 != k2:
            print("WARNING: JSON key sets differ (they should match LABEL_SCHEMA_KEYS + metadata).", file=sys.stderr)
            return 2
        print("OK: both payloads use the same top-level keys.")
        print("Expected clinical keys:", list(LABEL_SCHEMA_KEYS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
