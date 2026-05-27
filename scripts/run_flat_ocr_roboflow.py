#!/usr/bin/env python3
"""
Run Path A (PaddleOCR) on flat_label images from a Roboflow YOLO export.

Prerequisites:
  1. Download dataset: python scripts/download_roboflow_dataset.py
  2. .env with ROBOFLOW_* only needed for download; OCR uses local Paddle.

  export PYTHONPATH=src
  python scripts/run_flat_ocr_roboflow.py
  python scripts/run_flat_ocr_roboflow.py --dataset data/labeled/MedLabel-1 --limit 5
  python scripts/run_flat_ocr_roboflow.py --json --out data/processed/flat_ocr_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _find_dataset_root(base: Path) -> Path | None:
    """Locate a YOLO export root (contains data.yaml or train/images)."""
    if (base / "data.yaml").is_file() or (base / "train" / "images").is_dir():
        return base
    for child in sorted(base.iterdir()):
        if child.is_dir():
            found = _find_dataset_root(child)
            if found:
                return found
    return None


def _flat_class_id(dataset_root: Path, class_name: str) -> int:
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"No data.yaml under {dataset_root}")
    text = yaml_path.read_text(encoding="utf-8")
    names: list[str] = []
    in_names = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("names:"):
            in_names = True
            if "[" in s:
                # names: ['a', 'b']
                inner = s.split("[", 1)[1].split("]", 1)[0]
                names = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
                in_names = False
            continue
        if in_names and s.startswith("- "):
            names.append(s[2:].strip().strip("'\""))
        elif in_names and s and not s.startswith("#"):
            in_names = False
    if not names:
        raise ValueError(f"Could not parse class names from {yaml_path}")
    try:
        return names.index(class_name)
    except ValueError as e:
        raise ValueError(f"class {class_name!r} not in {names}") from e


def _label_has_class(label_path: Path, class_id: int) -> bool:
    if not label_path.is_file():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts and int(parts[0]) == class_id:
            return True
    return False


def iter_flat_images(dataset_root: Path, flat_class: str) -> Iterator[Path]:
    flat_id = _flat_class_id(dataset_root, flat_class)
    for split in ("train", "valid", "test"):
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            if _label_has_class(lbl, flat_id):
                yield img


def main() -> int:
    _setup_path()
    import cv2
    from vision.ocr import PathAOCR

    p = argparse.ArgumentParser(description="PaddleOCR on Roboflow flat_label images.")
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/labeled"),
        help="Roboflow download folder (default: data/labeled)",
    )
    p.add_argument(
        "--flat-class",
        default="flat_label",
        help="YOLO class name for flat packaging (default: flat_label)",
    )
    p.add_argument("--limit", type=int, default=0, help="Max images (0 = all)")
    p.add_argument("--no-preprocess", action="store_true")
    p.add_argument("--auto-rotate", action="store_true")
    p.add_argument("--json", action="store_true", help="Print JSON to stdout")
    p.add_argument("--out", type=Path, help="Write JSON results to this file")
    args = p.parse_args()

    root = _find_dataset_root(args.dataset.resolve())
    if root is None:
        print(
            f"error: no YOLO dataset under {args.dataset} (no data.yaml or train/images).",
            file=sys.stderr,
        )
        print(
            "Steps:\n"
            "  1. In Roboflow: open your project → Generate → create a dataset Version\n"
            "  2. python scripts/download_roboflow_dataset.py\n"
            "  3. python scripts/run_flat_ocr_roboflow.py --limit 5",
            file=sys.stderr,
        )
        return 1

    images = list(iter_flat_images(root, args.flat_class))
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(
            f"error: no images with class {args.flat_class!r} under {root}",
            file=sys.stderr,
        )
        return 1

    print(f"dataset: {root}", file=sys.stderr)
    print(f"flat_label images: {len(images)}", file=sys.stderr)

    ocr = PathAOCR()
    results: list[dict] = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"skip unreadable: {img_path}", file=sys.stderr)
            continue
        lines = ocr.run(img, preprocess=not args.no_preprocess, auto_page_rotate=args.auto_rotate)
        entry = {
            "image": str(img_path),
            "lines": lines,
            "raw_text": "\n".join(str(r["text"]) for r in lines),
        }
        results.append(entry)
        if not args.json and not args.out:
            print(f"=== {img_path.name} ===")
            for row in lines:
                print(f"{row['confidence']:.3f}\t{row['text']}")
            print()

    if args.json or args.out:
        payload = {"dataset_root": str(root), "flat_class": args.flat_class, "results": results}
        text = json.dumps(payload, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
            print(f"wrote {args.out}", file=sys.stderr)
        else:
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
