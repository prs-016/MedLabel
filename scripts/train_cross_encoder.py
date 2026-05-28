#!/usr/bin/env python3
"""
Fine-tune the MedLabel cross-encoder reranker.

Weak labels are built from FDA + DDInter chunks. Add gold pairs in
``data/training/cross_encoder_pairs.jsonl`` (one JSON object per line)::

  {"query": "acetaminophen ibuprofen interaction", "passage": "...", "label": 1.0}

Usage::

  export PYTHONPATH=src
  pip install sentence-transformers torch

  python scripts/train_cross_encoder.py
  python scripts/train_cross_encoder.py --epochs 3 --output models/cross_encoder_medlabel

After training, set in ``.env``::

  CROSS_ENCODER_MODEL=models/cross_encoder_medlabel
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _setup() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def main() -> int:
    root = _setup()
    from knowledge.training_data import (
        DEFAULT_MANUAL_PATH,
        generate_weak_supervision_examples,
        save_examples_jsonl,
    )

    p = argparse.ArgumentParser(description="Fine-tune cross-encoder for chunk reranking")
    p.add_argument(
        "--base-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HuggingFace cross-encoder base checkpoint",
    )
    p.add_argument(
        "--output",
        default=str(root / "models" / "cross_encoder_medlabel"),
        help="Directory to save fine-tuned model",
    )
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--negatives", type=int, default=2, help="Random negatives per positive")
    p.add_argument(
        "--save-dataset",
        type=Path,
        default=root / "data" / "training" / "generated_pairs.jsonl",
        help="Write generated training pairs for inspection",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    examples = generate_weak_supervision_examples(
        negatives_per_positive=args.negatives,
        seed=args.seed,
    )
    if len(examples) < 20:
        logger.error(
            "Need more training pairs (got %d). Run: python scripts/ingest_to_chromadb.py "
            "or add rows to %s",
            len(examples),
            DEFAULT_MANUAL_PATH,
        )
        return 1

    save_examples_jsonl(examples, args.save_dataset)
    logger.info("Saved dataset to %s", args.save_dataset)

    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader

    train_samples = [
        InputExample(texts=[query, passage], label=float(label))
        for query, passage, label in examples
    ]
    train_loader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)

    logger.info(
        "Training %s on %d pairs → %s",
        args.base_model,
        len(train_samples),
        args.output,
    )
    model = CrossEncoder(args.base_model, num_labels=1)
    warmup = max(10, len(train_loader) // 10)
    model.fit(
        train_dataloader=train_loader,
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        output_path=args.output,
        show_progress_bar=True,
    )

    logger.info("Done. Set CROSS_ENCODER_MODEL=%s in .env", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
