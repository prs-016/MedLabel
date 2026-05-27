#!/usr/bin/env python3
"""
CLI demo: image → OCR → interaction question.

  export PYTHONPATH=src
  python scripts/demo_cli.py -i data/raw/images/sample_ocr_test.png --flat \\
      --question "Can I take this with ibuprofen?"
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def _setup() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _setup()
    from agent.interaction_check import interaction_check
    from pipeline.scan import ocr_result_to_dict, resolve_scanned_drug, scan_image

    p = argparse.ArgumentParser(description="MedLabel scan + interaction demo")
    p.add_argument("-i", "--image", required=True, type=Path)
    p.add_argument("--flat", action="store_true", help="Flat label (Path A)")
    p.add_argument("--cylindrical", action="store_true", help="Bottle (Path B)")
    p.add_argument("--question", default="Can I take this with ibuprofen?")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if args.flat and args.cylindrical:
        print("Choose only one of --flat or --cylindrical", file=sys.stderr)
        return 1
    packaging = "cylindrical" if args.cylindrical else "flat"

    if not args.image.is_file():
        print(f"Not a file: {args.image}", file=sys.stderr)
        return 1

    ocr = scan_image(str(args.image), packaging)  # type: ignore[arg-type]
    scanned = resolve_scanned_drug(ocr)
    payload = ocr_result_to_dict(ocr, scanned_drug=scanned)

    result = interaction_check(args.question, scanned_drug=scanned)

    if args.json:
        out = {
            "scan": payload,
            "interaction": result.to_dict(),
        }
        print(json.dumps(out, indent=2))
    else:
        print("=== Scan ===")
        print(json.dumps(payload, indent=2))
        print()
        print(result.format_for_agent())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
