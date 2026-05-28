"""
Ingest FDA labels and DDInter rows into ChromaDB for reranked retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from api.ddinter import DDInterClient
from api.openfda import OpenFDAClient
from knowledge.vectorstore import MedLabelVectorStore

logger = logging.getLogger(__name__)

# Starter set for demos (expand as needed)
STARTER_DRUGS = [
    "acetaminophen",
    "ibuprofen",
    "aspirin",
    "loratadine",
    "diphenhydramine",
    "omeprazole",
    "metformin",
    "lisinopril",
    "amoxicillin",
    "sertraline",
]


def ddinter_to_chunks(client: Optional[DDInterClient] = None) -> list[dict[str, Any]]:
    """Convert DDInter CSV rows to text chunks with metadata."""
    dd = client or DDInterClient()
    dd.load()
    chunks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in dd.iter_all_records():
        text = (
            f"Drug interaction: {rec.drug_a} + {rec.drug_b}. "
            f"Level: {rec.level or 'unknown'}. "
            f"Mechanism: {rec.mechanism}. Management: {rec.management}"
        ).strip()
        key = f"{rec.drug_a}|{rec.drug_b}|{text[:80]}"
        if key in seen:
            continue
        seen.add(key)
        chunks.append(
            {
                "text": text,
                "metadata": {
                    "drug_name": rec.drug_a,
                    "drug_b": rec.drug_b,
                    "section_type": "drug_interaction",
                    "source": "ddinter",
                    "level": rec.level or "",
                },
            }
        )
    return chunks


def ingest_fda_drugs(
    drug_names: list[str],
    store: Optional[MedLabelVectorStore] = None,
    client: Optional[OpenFDAClient] = None,
) -> int:
    fda = client or OpenFDAClient()
    vs = store or MedLabelVectorStore()
    total = 0
    for name in drug_names:
        chunks = fda.get_label_chunks(name)
        if chunks:
            total += vs.add_chunks(chunks)
            logger.info("Ingested %d chunks for %s", len(chunks), name)
    return total


def ingest_ddinter(store: Optional[MedLabelVectorStore] = None) -> int:
    vs = store or MedLabelVectorStore()
    chunks = ddinter_to_chunks()
    if not chunks:
        logger.warning("No DDInter chunks to ingest (CSV missing?)")
        return 0
    return vs.add_chunks(chunks)


def ingest_starter_set(store: Optional[MedLabelVectorStore] = None) -> dict[str, int]:
    vs = store or MedLabelVectorStore()
    return {
        "fda": ingest_fda_drugs(STARTER_DRUGS, store=vs),
        "ddinter": ingest_ddinter(store=vs),
        "total_in_db": vs.count,
    }
