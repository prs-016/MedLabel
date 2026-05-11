#!/usr/bin/env python3
"""
Run Path A (preprocess + PaddleOCR) on image files from the command line.

Use this until the Streamlit scanner exists.

  export PYTHONPATH=src
  python scripts/run_path_a_ocr.py -i data/raw/images/sample_ocr_test.png
  python scripts/run_path_a_ocr.py -i path/to/your_real_label.jpg --json
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
    import cv2
    from vision.ocr import PathAOCR

    p = argparse.ArgumentParser(description="Path A OCR: read flat label images (CLI test harness).")
    p.add_argument("-i", "--input", required=True, type=Path, nargs="+", help="One or more image paths")
    p.add_argument("--no-preprocess", action="store_true", help="Skip CLAHE + adaptive threshold")
    p.add_argument("--auto-rotate", action="store_true", help="Correct 0/90/180/270 page rotation (heuristic)")
    p.add_argument(
        "--rotate-mode",
        choices=("heuristic", "ocr", "none"),
        default="heuristic",
        help="With --auto-rotate: how to pick angle (default heuristic; ocr is slower)",
    )
    p.add_argument("--min-confidence", type=float, default=0.0, help="Drop lines below this confidence")
    p.add_argument("--json", action="store_true", help="Print JSON array instead of plain lines")
    args = p.parse_args()

    ocr = PathAOCR()
    rotate = args.auto_rotate
    mode = args.rotate_mode if rotate else "none"

    all_out: list[dict] = []
    for path in args.input:
        if not path.is_file():
            print(f"skip (not a file): {path}", file=sys.stderr)
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"skip (unreadable): {path}", file=sys.stderr)
            continue
        lines = ocr.run(
            img,
            preprocess=not args.no_preprocess,
            auto_page_rotate=rotate,
            page_rotate_mode=mode,
            min_confidence=args.min_confidence,
        )
        if args.json:
            all_out.append({"path": str(path), "lines": lines})
        else:
            print(f"=== {path} ===")
            for row in lines:
                print(f"{row['confidence']:.3f}\t{row['text']}")
            print()

    if args.json and all_out:
        print(json.dumps(all_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
