"""
Build (query, passage, label) pairs for cross-encoder fine-tuning.

Uses weak supervision from FDA label chunks + DDInter rows. For better quality,
add hand-labeled pairs in ``data/training/cross_encoder_pairs.jsonl``.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

from api.ddinter import DDInterClient
from api.openfda import OpenFDAClient
from knowledge.ingest import STARTER_DRUGS, ddinter_to_chunks

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUAL_PATH = ROOT / "data" / "training" / "cross_encoder_pairs.jsonl"


def _interaction_query(drug_a: str, drug_b: str) -> str:
    return f"{drug_a} {drug_b} drug drug interaction contraindication concomitant use"


def _adverse_query(drug: str) -> str:
    return f"{drug} adverse reaction side effect toxicity warning"


def _vector_query(drug: str, section: str) -> str:
    return f"{drug} {section.replace('_', ' ')}"


def load_manual_pairs(path: Path = DEFAULT_MANUAL_PATH) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_fda_chunks(drug_names: Optional[list[str]] = None) -> list[dict[str, Any]]:
    fda = OpenFDAClient()
    chunks: list[dict[str, Any]] = []
    for name in drug_names or STARTER_DRUGS:
        for ch in fda.get_label_chunks(name):
            meta = ch.get("metadata") or {}
            meta.setdefault("drug_name", name)
            chunks.append(ch)
    return chunks


def generate_weak_supervision_examples(
    *,
    drug_names: Optional[list[str]] = None,
    negatives_per_positive: int = 2,
    seed: int = 42,
) -> list[tuple[str, str, float]]:
    """
  Returns list of (query, passage_text, label) with label 1.0 or 0.0.
    """
    random.seed(seed)
    fda_chunks = build_fda_chunks(drug_names)
    ddinter_chunks = ddinter_to_chunks(DDInterClient())

    all_chunks = fda_chunks + ddinter_chunks
    if len(all_chunks) < 4:
        logger.warning("Very few chunks for training — ingest more drugs first.")

    examples: list[tuple[str, str, float]] = []

    def add_pair(query: str, chunk: dict[str, Any], label: float) -> None:
        text = (chunk.get("text") or "").strip()
        # Cross-encoder max length ~512 tokens — keep passages short
        if len(text) > 2000:
            text = text[:2000] + "…"
        if text:
            examples.append((query, text, label))

    def random_negative(exclude_idx: int) -> dict[str, Any]:
        idx = exclude_idx
        while idx == exclude_idx and len(all_chunks) > 1:
            idx = random.randint(0, len(all_chunks) - 1)
        return all_chunks[idx]

    # DDInter interaction positives
    for i, ch in enumerate(ddinter_chunks):
        meta = ch.get("metadata") or {}
        da = meta.get("drug_name", "")
        db = meta.get("drug_b", "")
        if da and db:
            q = _interaction_query(da, db)
            add_pair(q, ch, 1.0)
            for _ in range(negatives_per_positive):
                add_pair(q, random_negative(i), 0.0)

    # FDA section-aligned positives
    for i, ch in enumerate(fda_chunks):
        meta = ch.get("metadata") or {}
        drug = meta.get("drug_name", "")
        section = meta.get("section_type", "")
        if not drug or not section:
            continue

        if section == "drug_interactions":
            q = _interaction_query(drug, "other medicines")
            add_pair(q, ch, 1.0)
        elif section == "adverse_reactions":
            q = _adverse_query(drug)
            add_pair(q, ch, 1.0)
        else:
            q = _vector_query(drug, section)
            add_pair(q, ch, 1.0)

        for _ in range(negatives_per_positive):
            neg = random_negative(i)
            add_pair(q, neg, 0.0)

    # Manual gold pairs
    for row in load_manual_pairs():
        examples.append(
            (
                row["query"],
                row["passage"],
                float(row.get("label", 1.0)),
            )
        )

    n_pos = sum(1 for _, _, label in examples if label >= 0.5)
    logger.info("Generated %d training pairs (%d positive)", len(examples), n_pos)
    return examples


def save_examples_jsonl(
    examples: list[tuple[str, str, float]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for query, passage, label in examples:
            f.write(
                json.dumps(
                    {"query": query, "passage": passage, "label": label},
                    ensure_ascii=False,
                )
                + "\n"
            )
