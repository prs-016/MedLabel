import re
import time
import logging
from dataclasses import dataclass
from typing import Optional
import requests

logger = logging.getLogger(__name__)

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"


@dataclass
class ResolvedDrug:
    rxcui:        str = ""
    generic_name: str = ""
    brand_name:   str = ""
    matched_name: str = ""
    original_ocr: str = ""

    @property
    def found(self) -> bool:
        return bool(self.rxcui)


class RxNormClient:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        url = f"{RXNORM_BASE}/{endpoint}"
        for attempt in range(1, 4):
            try:
                resp = self.session.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 404:
                    return None
                time.sleep(2 ** attempt)
            except requests.RequestException as exc:
                logger.error("RxNorm error (attempt %d): %s", attempt, exc)
                if attempt < 3:
                    time.sleep(2 ** attempt)
        return None

    def get_rxcui(self, drug_name: str) -> Optional[str]:
        data = self._get("rxcui.json", params={"name": drug_name, "allSourcesFlag": 1})
        if not data:
            return None
        rxcui = data.get("idGroup", {}).get("rxnormId", [None])[0]
        return rxcui or None

    def get_generic_name(self, drug_name: str) -> Optional[str]:
        rxcui = self.get_rxcui(drug_name)
        if not rxcui:
            return None
        data = self._get(f"rxcui/{rxcui}/related.json", params={"tty": "IN"})
        if not data:
            return None
        for group in data.get("relatedGroup", {}).get("conceptGroup", []):
            for concept in group.get("conceptProperties", []):
                name = concept.get("name", "")
                if name:
                    return name.lower()
        return None

    def resolve_ocr_string(self, raw_ocr: str) -> ResolvedDrug:
        cleaned = self._clean_ocr(raw_ocr)
        logger.info("OCR resolve: '%s' → cleaned: '%s'", raw_ocr, cleaned)

        # Strategy 1: try the full cleaned string
        result = self._try_resolve(cleaned, raw_ocr)
        if result.found:
            return result

        # Strategy 2: try each word token, longest first
        for token in self._meaningful_tokens(cleaned):
            result = self._try_resolve(token, raw_ocr)
            if result.found:
                return result

        # Strategy 3: fuzzy/approximate search
        result = self._fuzzy_resolve(cleaned, raw_ocr)
        if result.found:
            return result

        logger.warning("Could not resolve OCR string: '%s'", raw_ocr)
        return ResolvedDrug(original_ocr=raw_ocr)

    @staticmethod
    def _clean_ocr(raw: str) -> str:
        text = raw.strip()
        substitutions = [
            (r'(?<=[a-zA-Z])0(?=[a-zA-Z])', 'o'),
            (r'(?<=[a-zA-Z])1(?=[a-zA-Z])', 'i'),
            (r'\|', 'I'),
            (r'\+', 't'),
            (r'@',  'a'),
            (r'\$', 'S'),
        ]
        for pattern, replacement in substitutions:
            text = re.sub(pattern, replacement, text)
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def _meaningful_tokens(text: str) -> list[str]:
        dosage = re.compile(
            r'^\d+(\.\d+)?\s*(mg|mcg|ml|g|%|iu|tabs?|caps?)s?$', re.IGNORECASE
        )
        tokens = [
            t for t in text.split()
            if len(t) >= 3 and not t.isdigit() and not dosage.match(t)
        ]
        return sorted(tokens, key=len, reverse=True)

    def _try_resolve(self, name: str, original: str) -> ResolvedDrug:
        rxcui = self.get_rxcui(name)
        if not rxcui:
            return ResolvedDrug(original_ocr=original)
        generic = self.get_generic_name(name) or name.lower()
        return ResolvedDrug(rxcui=rxcui, generic_name=generic,
                            matched_name=name, original_ocr=original)

    def _fuzzy_resolve(self, cleaned: str, original: str) -> ResolvedDrug:
        data = self._get("approximateTerm.json", params={"term": cleaned, "maxEntries": 1})
        if not data:
            return ResolvedDrug(original_ocr=original)
        candidates = data.get("approximateGroup", {}).get("candidate", [])
        if not candidates:
            return ResolvedDrug(original_ocr=original)
        rxcui = candidates[0].get("rxcui", "")
        if not rxcui:
            return ResolvedDrug(original_ocr=original)
        name_data = self._get(f"rxcui/{rxcui}/properties.json")
        matched = name_data.get("properties", {}).get("name", cleaned) if name_data else cleaned
        generic = self.get_generic_name(matched) or matched.lower()
        return ResolvedDrug(rxcui=rxcui, generic_name=generic,
                            matched_name=matched, original_ocr=original)