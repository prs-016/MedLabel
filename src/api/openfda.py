import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
import requests

logger = logging.getLogger(__name__)

FDA_LABEL_BASE = "https://api.fda.gov/drug/label.json"
FDA_EVENT_BASE = "https://api.fda.gov/drug/event.json"

LABEL_SECTIONS = [
    "dosage_and_administration",
    "warnings",
    "drug_interactions",
    "indications_and_usage",
    "active_ingredient",
    "inactive_ingredient",
    "purpose",
    "keep_out_of_reach_of_children",
    "stop_use",
    "ask_doctor",
    "ask_doctor_or_pharmacist",
    "pregnancy_or_breast_feeding",
    "overdosage",
    "adverse_reactions",
]


@dataclass
class DrugLabel:
    brand_name:                    str = ""
    generic_name:                  str = ""
    rxcui:                         str = ""
    dosage_and_administration:     list[str] = field(default_factory=list)
    warnings:                      list[str] = field(default_factory=list)
    drug_interactions:             list[str] = field(default_factory=list)
    indications_and_usage:         list[str] = field(default_factory=list)
    active_ingredient:             list[str] = field(default_factory=list)
    inactive_ingredient:           list[str] = field(default_factory=list)
    purpose:                       list[str] = field(default_factory=list)
    keep_out_of_reach_of_children: list[str] = field(default_factory=list)
    stop_use:                      list[str] = field(default_factory=list)
    ask_doctor:                    list[str] = field(default_factory=list)
    ask_doctor_or_pharmacist:      list[str] = field(default_factory=list)
    pregnancy_or_breast_feeding:   list[str] = field(default_factory=list)
    overdosage:                    list[str] = field(default_factory=list)
    adverse_reactions:             list[str] = field(default_factory=list)

    def to_chunks(self) -> list[dict]:
        chunks = []
        for section in LABEL_SECTIONS:
            for text in getattr(self, section, []):
                if text.strip():
                    chunks.append({
                        "text": text.strip(),
                        "metadata": {
                            "drug_name":    self.generic_name or self.brand_name,
                            "brand_name":   self.brand_name,
                            "rxcui":        self.rxcui,
                            "section_type": section,
                            "source":       "fda_label",
                        }
                    })
        return chunks


class OpenFDAClient:

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENFDA_API_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MedLabel/1.0"})

    def _build_params(self, extra: dict) -> dict:
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key
        params.update(extra)
        return params

    def _get(self, url: str, params: dict, retries: int = 3) -> Optional[dict]:
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 404:
                    return None
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                return None
            except requests.RequestException as exc:
                logger.error("openFDA error (attempt %d): %s", attempt, exc)
                if attempt < retries:
                    time.sleep(2 ** attempt)
        return None

    def get_label(self, drug_name: str) -> Optional[DrugLabel]:
        searches = [
            f'openfda.generic_name:"{drug_name}"',
            f'openfda.brand_name:"{drug_name}"',
        ]
        for search in searches:
            params = self._build_params({"search": search, "limit": 1})
            data = self._get(FDA_LABEL_BASE, params)
            if data and data.get("results"):
                return self._parse_label(data["results"][0])
        logger.warning("No FDA label found for '%s'", drug_name)
        return None

    def get_label_chunks(self, drug_name: str) -> list[dict]:
        label = self.get_label(drug_name)
        return label.to_chunks() if label else []

    def get_adverse_events(self, drug_name: str, limit: int = 10) -> list[dict]:
        """Top FAERS reaction counts for a drug (tries several openFDA search fields)."""
        name = drug_name.strip()
        if not name:
            return []

        searches = [
            f'patient.drug.openfda.generic_name:"{name}"',
            f'patient.drug.openfda.brand_name:"{name}"',
            f'patient.drug.medicinalproduct:"{name}"',
        ]
        for search in searches:
            params = self._build_params({
                "search": search,
                "count": "patient.reaction.reactionmeddrapt.exact",
                "limit": limit,
            })
            data = self._get(FDA_EVENT_BASE, params)
            if data and data.get("results"):
                return [
                    {"reaction": r["term"], "count": r["count"]}
                    for r in data["results"]
                ]
        return []

    def _parse_label(self, raw: dict) -> DrugLabel:
        meta = raw.get("openfda", {})
        label = DrugLabel(
            brand_name=self._first(meta.get("brand_name", [])),
            generic_name=self._first(meta.get("generic_name", [])),
        )
        for section in LABEL_SECTIONS:
            value = raw.get(section, [])
            if isinstance(value, str):
                value = [value]
            setattr(label, section, value)
        return label

    @staticmethod
    def _first(lst: list, default: str = "") -> str:
        return lst[0] if lst else default