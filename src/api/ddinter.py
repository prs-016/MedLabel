"""
DDInter drug–drug interaction lookup from local CSV files.

Place files under ``data/ddinter/`` (see README for download links).
Supports common column layouts from DDInter / Kaggle mirrors.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DDINTER_DIR = Path(__file__).resolve().parents[2] / "data" / "ddinter"

# Map normalized header → canonical field
_HEADER_ALIASES: dict[str, str] = {
    "drug_a": "drug_a",
    "drug1": "drug_a",
    "drug 1": "drug_a",
    "drug1name": "drug_a",
    "drug_b": "drug_b",
    "drug2": "drug_b",
    "drug 2": "drug_b",
    "drug2name": "drug_b",
    "level": "level",
    "severity": "level",
    "interaction_level": "level",
    "ddinterid_a": "id_a",
    "ddinterid_b": "id_b",
    "mechanism": "mechanism",
    "management": "management",
}


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _pair_key(a: str, b: str) -> tuple[str, str]:
    na, nb = _normalize_name(a), _normalize_name(b)
    return (na, nb) if na <= nb else (nb, na)


@dataclass
class DDIRecord:
    drug_a: str
    drug_b: str
    level: str = ""
    mechanism: str = ""
    management: str = ""
    source_file: str = ""

    def to_dict(self) -> dict:
        return {
            "drug_a": self.drug_a,
            "drug_b": self.drug_b,
            "level": self.level,
            "mechanism": self.mechanism,
            "management": self.management,
            "source": "ddinter",
            "source_file": self.source_file,
        }


@dataclass
class DDInterClient:
    """In-memory index of DDInter CSV rows."""

    data_dir: Path = field(default_factory=lambda: DEFAULT_DDINTER_DIR)
    _pair_index: dict[tuple[str, str], list[DDIRecord]] = field(default_factory=dict, repr=False)
    _name_tokens: dict[str, set[tuple[str, str]]] = field(default_factory=dict, repr=False)
    loaded: bool = False
    row_count: int = 0

    def load(self, force: bool = False) -> int:
        if self.loaded and not force:
            return self.row_count
        self._pair_index.clear()
        self._name_tokens.clear()
        self.row_count = 0

        if not self.data_dir.is_dir():
            logger.warning("DDInter directory missing: %s", self.data_dir)
            self.loaded = True
            return 0

        for path in sorted(self.data_dir.glob("*.csv")):
            self._load_csv(path)

        self.loaded = True
        logger.info("DDInter loaded %d rows from %s", self.row_count, self.data_dir)
        return self.row_count

    def _load_csv(self, path: Path) -> None:
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return
                colmap = self._map_columns(reader.fieldnames)
                if "drug_a" not in colmap or "drug_b" not in colmap:
                    logger.warning("Skipping %s — no drug_a/drug_b columns", path.name)
                    return
                for row in reader:
                    da = (row.get(colmap["drug_a"]) or "").strip()
                    db = (row.get(colmap["drug_b"]) or "").strip()
                    if not da or not db:
                        continue
                    rec = DDIRecord(
                        drug_a=da,
                        drug_b=db,
                        level=(row.get(colmap.get("level", ""), "") or "").strip(),
                        mechanism=(row.get(colmap.get("mechanism", ""), "") or "").strip(),
                        management=(row.get(colmap.get("management", ""), "") or "").strip(),
                        source_file=path.name,
                    )
                    key = _pair_key(da, db)
                    self._pair_index.setdefault(key, []).append(rec)
                    for name in (da, db):
                        nk = _normalize_name(name)
                        self._name_tokens.setdefault(nk, set()).add(key)
                    self.row_count += 1
        except OSError as e:
            logger.error("Failed to read %s: %s", path, e)

    @staticmethod
    def _map_columns(fieldnames: list[str]) -> dict[str, str]:
        colmap: dict[str, str] = {}
        for raw in fieldnames:
            norm = raw.strip().lower().replace(" ", "_")
            key = _HEADER_ALIASES.get(norm)
            if key:
                colmap[key] = raw
        return colmap

    def lookup_pair(self, drug_a: str, drug_b: str) -> list[DDIRecord]:
        self.load()
        key = _pair_key(drug_a, drug_b)
        return list(self._pair_index.get(key, []))

    def lookup_involving(self, drug_name: str, limit: int = 20) -> list[DDIRecord]:
        """All interactions where ``drug_name`` matches either side (exact normalized name)."""
        self.load()
        nk = _normalize_name(drug_name)
        keys = self._name_tokens.get(nk, set())
        out: list[DDIRecord] = []
        for key in keys:
            out.extend(self._pair_index.get(key, []))
        return out[:limit]

    def has_data(self) -> bool:
        self.load()
        return self.row_count > 0


# Built-in demo pairs when CSVs are not downloaded yet (common OTC examples)
_DEMO_PAIRS: list[DDIRecord] = [
    DDIRecord(
        drug_a="ibuprofen",
        drug_b="acetaminophen",
        level="Moderate",
        mechanism="Combined use may increase risk of gastrointestinal bleeding; monitor.",
        management="Use lowest effective doses; consult pharmacist if used together frequently.",
        source_file="demo",
    ),
    DDIRecord(
        drug_a="warfarin",
        drug_b="ibuprofen",
        level="Major",
        mechanism="NSAIDs may increase anticoagulant effect and bleeding risk.",
        management="Avoid combination unless closely monitored by clinician.",
        source_file="demo",
    ),
]


def get_demo_interactions(drug_a: str, drug_b: str) -> list[DDIRecord]:
    key = _pair_key(drug_a, drug_b)
    na, nb = key
    out = []
    for rec in _DEMO_PAIRS:
        ka, kb = _pair_key(rec.drug_a, rec.drug_b)
        if (ka, kb) == (na, nb):
            out.append(rec)
    return out
