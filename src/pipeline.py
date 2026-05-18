"""
src/pipeline.py
================
MedLabelPipeline — RxNorm → OpenFDA → DDInter → ChromaDB
"""

import uuid
import time
import logging
from typing import Optional

from vision.ocr         import OCRResult
from api.rxnorm         import RxNormClient
from api.openfda        import OpenFDAClient
from api.ddinter        import DDInterClient
from knowledge.embedder import DrugEmbedder

logger = logging.getLogger(__name__)

FIFTY_DRUGS = [
    "warfarin", "metformin", "lisinopril", "atorvastatin", "amoxicillin",
    "metoprolol", "omeprazole", "amlodipine", "sertraline", "levothyroxine",
    "clopidogrel", "furosemide", "gabapentin", "hydrochlorothiazide", "fluoxetine",
    "alprazolam", "prednisone", "ciprofloxacin", "azithromycin", "losartan",
    "escitalopram", "tramadol", "clonazepam", "simvastatin", "pantoprazole",
    "carvedilol", "montelukast", "tamsulosin", "rosuvastatin", "duloxetine",
    "quetiapine", "albuterol", "bupropion", "spironolactone", "venlafaxine",
    "zolpidem", "trazodone", "meloxicam", "naproxen", "ibuprofen",
    "celecoxib", "rivaroxaban", "apixaban", "methotrexate", "cyclosporine",
    "tacrolimus", "digoxin", "phenytoin", "carbamazepine", "lithium",
]


class MedLabelPipeline:

    def __init__(self, db_path: str = "./drug_db"):
        self.rxnorm   = RxNormClient()
        self.fda      = OpenFDAClient()
        self.ddinter  = DDInterClient()
        self.embedder = DrugEmbedder(db_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def process_ocr_result(self, ocr: OCRResult) -> dict:
        resolved = self.rxnorm.resolve_ocr_string(ocr.drug_name)
        if not resolved.found:
            return {"error": f"Could not resolve '{ocr.drug_name}' via RxNorm"}
        drug_name = resolved.generic_name
        chunks    = self._build_fda_chunks(drug_name, resolved.rxcui)
        n         = self.embedder.upsert_chunks(chunks)
        return {"drug_name": drug_name, "rxcui": resolved.rxcui, "chunks_added": n}

    def process_drug_list(
        self,
        drug_names: list[str],
        session_id: Optional[str] = None,
    ) -> dict:
        session_id = session_id or uuid.uuid4().hex

        canonical, rxcui_map = self._resolve_all(drug_names)
        if not canonical:
            return {"error": "No drugs could be resolved via RxNorm", "session_id": session_id}

        fda_chunks = []
        for name in canonical:
            fda_chunks.extend(self._build_fda_chunks(name, rxcui_map.get(name, "")))

        ddi_docs = self._build_ddi_chunks(canonical, session_id)

        n_fda = self.embedder.upsert_chunks(fda_chunks)
        n_ddi = self.embedder.upsert_chunks(ddi_docs)

        logger.info(
            "process_drug_list session=%s  fda=%d  ddi=%d  total=%d",
            session_id, n_fda, n_ddi, n_fda + n_ddi,
        )

        return {
            "session_id":     session_id,
            "canonical":      canonical,
            "fda_chunks":     n_fda,
            "ddi_docs":       n_ddi,
            "total_upserted": n_fda + n_ddi,
        }

    def seed_database(self, drug_list: list[str] = FIFTY_DRUGS, delay: float = 0.3):
        """Seed ChromaDB with FDA label data for all drugs."""
        logger.info("Seeding %d drugs...", len(drug_list))
        for drug in drug_list:
            try:
                resolved  = self.rxnorm.resolve_ocr_string(drug)
                rxcui     = resolved.rxcui        if resolved.found else ""
                drug_name = resolved.generic_name if resolved.found else drug
                chunks    = self._build_fda_chunks(drug_name, rxcui)
                self.embedder.upsert_chunks(chunks)
                logger.info("✓ %s (%d chunks)", drug_name, len(chunks))
            except Exception as exc:
                logger.error("✗ %s — %s", drug, exc)
            time.sleep(delay)

    def seed_interactions(self, drug_list: list[str] = FIFTY_DRUGS, delay: float = 1.0):
        """
        Run DDInter for all 50 drugs in batches of 5 and store in ChromaDB.
        Splits the list into consecutive batches of 5, so every drug gets
        checked against its neighbors.
        """
        batches = [drug_list[i:i+5] for i in range(0, len(drug_list), 5)]
        logger.info("Running DDInter for %d batches...", len(batches))

        total_ddi = 0
        for idx, batch in enumerate(batches):
            if len(batch) < 2:
                continue
            print(f"Batch {idx+1}/{len(batches)}: {batch}")
            try:
                result    = self.process_drug_list(batch)
                n_ddi     = result.get("ddi_docs", 0)
                total_ddi += n_ddi
                print(f"  → {n_ddi} DDI docs added")
            except Exception as exc:
                print(f"  ✗ batch failed: {exc}")
            time.sleep(delay)

        print(f"\nDone. Total DDI docs added: {total_ddi}")

    def query(
        self,
        text: str,
        n_results: int = 5,
        min_severity: Optional[int] = None,
    ) -> list[dict]:
        where = None
        if min_severity is not None:
            where = {"severity_score": {"$gte": min_severity}}
        return self.embedder.query(text, n_results=n_results, where=where)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_all(self, drug_names: list[str]) -> tuple[list[str], dict[str, str]]:
        canonical, rxcui_map = [], {}
        for name in drug_names:
            resolved = self.rxnorm.resolve_ocr_string(name)
            if resolved.found:
                canonical.append(resolved.generic_name)
                rxcui_map[resolved.generic_name] = resolved.rxcui
            else:
                logger.warning("Could not resolve '%s' — skipping for DDInter.", name)
        return canonical, rxcui_map

    def _build_fda_chunks(self, drug_name: str, rxcui: str) -> list[dict]:
        chunks = self.fda.get_label_chunks(drug_name)
        for chunk in chunks:
            chunk["metadata"]["rxcui"] = rxcui
        return chunks

    def _build_ddi_chunks(self, canonical_names: list[str], session_id: str) -> list[dict]:
        if len(canonical_names) < 2:
            return []

        if len(canonical_names) <= 5:
            return self.ddinter.check_interactions_for_session(
                canonical_names, session_id
            )

        seen_ids: set[str] = set()
        all_docs: list[dict] = []
        for i in range(len(canonical_names) - 1):
            batch = canonical_names[i : i + 5]
            docs  = self.ddinter.check_interactions_for_session(
                batch, f"{session_id}-b{i}"
            )
            for doc in docs:
                if doc["id"] not in seen_ids:
                    seen_ids.add(doc["id"])
                    doc["metadata"]["session_id"] = session_id
                    all_docs.append(doc)

        return all_docs