"""
Evaluate cross-encoder rerankers on held-out query/chunk pairs.

Usage (from repo root):
    PYTHONPATH=src python src/knowledge/eval_rerankers.py
    PYTHONPATH=src python src/knowledge/eval_rerankers.py --compare-base
    PYTHONPATH=src python src/knowledge/eval_rerankers.py --holdout 0.2 --seed 42

Metrics per model:
  - accuracy     : score sign matches label (positive label → score > 0)
  - pos/neg mean : average score on label=1 vs label=0
  - pairwise     : among (query, pos_chunk, neg_chunk) triples from the same
                   source chunk, fraction where pos_score > neg_score
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sentence_transformers import CrossEncoder

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MODELS = {
    "vector_search": {
        "path": os.path.join(ROOT_DIR, "models", "vector_search_reranker"),
        "data": os.path.join(ROOT_DIR, "data", "vector_search_pairs.json"),
    },
    "adverse_events": {
        "path": os.path.join(ROOT_DIR, "models", "adverse_events_reranker"),
        "data": os.path.join(ROOT_DIR, "data", "adverse_events_pairs.json"),
    },
    "interaction_check": {
        "path": os.path.join(ROOT_DIR, "models", "medlabel_reranker"),
        "data": (
            os.path.join(ROOT_DIR, "data", "interaction_pairs.json")
            if os.path.isfile(os.path.join(ROOT_DIR, "data", "interaction_pairs.json"))
            else os.path.join(ROOT_DIR, "data", "training_pairs.json")
        ),
    },
}


@dataclass
class EvalResult:
    name: str
    n_pairs: int
    accuracy: float
    pos_mean: float
    neg_mean: float
    separation: float
    pairwise_acc: float
    n_pairwise: int


def _load_pairs(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _split_holdout(
    pairs: list[dict], holdout: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Hold out by chunk text so the same passage cannot leak train→test."""
    by_chunk: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_chunk[p["chunk"][:200]].append(p)

    keys = list(by_chunk.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_test = max(1, int(len(keys) * holdout))
    test_keys = set(keys[:n_test])

    train, test = [], []
    for key, group in by_chunk.items():
        (test if key in test_keys else train).extend(group)
    return train, test


def _pairwise_accuracy(pairs: list[dict], scores: np.ndarray) -> tuple[float, int]:
    """
    Group rows by chunk; when a chunk has both label 1 and 0 queries,
    check whether every positive query scores higher than every negative.
    """
    by_chunk: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for p, score in zip(pairs, scores):
        key = p["chunk"][:200]
        by_chunk[key].append((p["query"], float(p["label"]), float(score)))

    wins, total = 0, 0
    for rows in by_chunk.values():
        pos = [(q, s) for q, lab, s in rows if lab == 1.0]
        neg = [(q, s) for q, lab, s in rows if lab == 0.0]
        if not pos or not neg:
            continue
        for _, ps in pos:
            for _, ns in neg:
                total += 1
                if ps > ns:
                    wins += 1
    return (wins / total if total else 0.0), total


def evaluate_model(
    name: str,
    model_path: str,
    pairs: list[dict],
) -> EvalResult:
    if not pairs:
        raise ValueError(f"No evaluation pairs for {name}")

    if os.path.isdir(model_path):
        model = CrossEncoder(model_path)
    elif model_path == BASE_MODEL or "/" in model_path:
        model = CrossEncoder(model_path)
    else:
        print(f"  [{name}] model not found at {model_path}, using {BASE_MODEL}")
        model = CrossEncoder(BASE_MODEL)

    inputs = [[p["query"], p["chunk"]] for p in pairs]
    scores = np.asarray(model.predict(inputs), dtype=float)
    labels = np.asarray([float(p["label"]) for p in pairs])

    preds = (scores > 0).astype(float)
    accuracy = float((preds == labels).mean())
    pos_mean = float(scores[labels == 1].mean()) if (labels == 1).any() else 0.0
    neg_mean = float(scores[labels == 0].mean()) if (labels == 0).any() else 0.0
    pw_acc, n_pw = _pairwise_accuracy(pairs, scores)

    return EvalResult(
        name=name,
        n_pairs=len(pairs),
        accuracy=accuracy,
        pos_mean=pos_mean,
        neg_mean=neg_mean,
        separation=pos_mean - neg_mean,
        pairwise_acc=pw_acc,
        n_pairwise=n_pw,
    )


def _print_result(r: EvalResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {r.name}")
    print(f"{'=' * 60}")
    print(f"  pairs evaluated : {r.n_pairs}")
    print(f"  accuracy        : {r.accuracy:.1%}  (score>0 ↔ label=1)")
    print(f"  pos mean score  : {r.pos_mean:+.3f}")
    print(f"  neg mean score  : {r.neg_mean:+.3f}")
    print(f"  separation      : {r.separation:.3f}")
    print(f"  pairwise acc    : {r.pairwise_acc:.1%}  ({r.n_pairwise} query pairs)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reranker models")
    parser.add_argument(
        "--holdout", type=float, default=0.2,
        help="Fraction of unique chunks held out for eval (default 0.2)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compare-base", action="store_true",
        help="Also score each dataset with the untuned ms-marco base model",
    )
    parser.add_argument(
        "--functions", nargs="*",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Which functions to evaluate (default: all)",
    )
    args = parser.parse_args()

    print(f"Holdout: {args.holdout:.0%} of chunks by text (seed={args.seed})\n")

    for func in args.functions:
        cfg = MODELS[func]
        if not os.path.isfile(cfg["data"]):
            print(f"SKIP {func}: missing {cfg['data']}")
            continue

        all_pairs = _load_pairs(cfg["data"])
        _, test_pairs = _split_holdout(all_pairs, args.holdout, args.seed)
        print(f"--- {func} ---")
        print(f"  total pairs: {len(all_pairs)}  |  holdout test: {len(test_pairs)}")

        r = evaluate_model(func, cfg["path"], test_pairs)
        _print_result(r)

        if args.compare_base:
            r_base = evaluate_model(f"{func} (base CE)", BASE_MODEL, test_pairs)
            _print_result(r_base)
            delta = r.accuracy - r_base.accuracy
            print(f"\n  fine-tuned vs base accuracy delta: {delta:+.1%}")

    print("\nDone. Regenerate training JSON on medlabel_db before retraining if")
    print("holdout scores are weak or base CE matches fine-tuned.")


if __name__ == "__main__":
    main()
