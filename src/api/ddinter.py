"""
src/api/ddinter.py
==================
DDInter 2.0 web client — replaces the old CSV-based DDInterClient.
"""

import re
import logging
import hashlib
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

BASE    = "https://ddinter2.scbdd.com"
HEADERS = {
    "User-Agent":       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/148.0.0.0 Safari/537.36",
    "Referer":          f"{BASE}/inter-checker/",
    "X-Requested-With": "XMLHttpRequest",
    "Accept":           "application/json, text/javascript, */*; q=0.01",
}

SEVERITY_MAP = {"Minor": 1, "Moderate": 2, "Major": 3}


@dataclass
class DDInterResult:
    drug_a:           str = ""
    drug_b:           str = ""
    interaction_type: str = ""
    severity_label:   str = ""
    severity_score:   int = 0
    interaction_text: str = ""
    management_text:  str = ""
    detail_url:       str = ""

    def to_chunk(self, anchor_drug: str) -> dict:
        other = self.drug_b if self.drug_a.lower() == anchor_drug.lower() else self.drug_a
        text = (
            f"{self.drug_a} and {self.drug_b} interact. "
            f"Severity: {self.severity_label}. "
            f"{self.interaction_text} "
            f"Management: {self.management_text}"
        ).strip()
        return {
            "text": text,
            "metadata": {
                "drug_name":        anchor_drug,
                "interacting_drug": other,
                "interaction_type": self.interaction_type,
                "severity_label":   self.severity_label,
                "severity_score":   self.severity_score,
                "section_type":     "drug_drug_interaction",
                "source":           "ddinter2_web",
                "detail_url":       self.detail_url,
            },
        }


class DDInterClient:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._cache: dict[str, str] = {}

    def get_interactions(self, drug_name: str) -> list[DDInterResult]:
        logger.warning(
            "get_interactions('%s') called in single-drug mode. "
            "Use check_interactions_for_session() with the full drug list.", drug_name
        )
        return []

    def get_max_severity(self, drug_name: str) -> int:
        return max((r.severity_score for r in self.get_interactions(drug_name)), default=0)

    def get_chunks(self, drug_name: str) -> list[dict]:
        return [r.to_chunk(drug_name) for r in self.get_interactions(drug_name)]

    def check_interactions_for_session(
        self,
        canonical_names: list[str],
        session_id: str,
    ) -> list[dict]:
        if len(canonical_names) < 2:
            logger.info("DDInter: fewer than 2 drugs — skipping.")
            return []
        if len(canonical_names) > 5:
            logger.warning("DDInter: truncating to first 5 drugs.")
            canonical_names = canonical_names[:5]
        raw = self._check_interactions(canonical_names)
        return self._to_chroma_documents(raw, session_id)

    def _search_drug(self, query: str) -> list[dict]:
        try:
            r = self.session.get(f"{BASE}/check-datasource/{query}/", timeout=10)
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as exc:
            logger.error("DDInter search error for '%s': %s", query, exc)
            return []

    def _resolve_name(self, name: str) -> Optional[str]:
        if name in self._cache:
            return self._cache[name]
        results = self._search_drug(name)
        if not results:
            logger.warning("DDInter: '%s' not found.", name)
            return None
        for drug in results:
            if drug["name"].lower() == name.lower():
                self._cache[name] = drug["internalID"]
                return drug["internalID"]
        first = results[0]
        logger.info("DDInter: '%s' → using '%s' (%s)", name, first["name"], first["internalID"])
        self._cache[name] = first["internalID"]
        return first["internalID"]

    def _check_interactions(self, drug_names: list[str]) -> dict:
        ids = [did for name in drug_names if (did := self._resolve_name(name))]
        if len(ids) < 2:
            logger.warning("DDInter: could not resolve enough drug names.")
            return {}
        url = f"{BASE}/checker/result/{'-'.join(ids)}/"
        return self._parse_result_page(url)

    def _parse_result_page(self, url: str) -> dict:
        try:
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
        except Exception as exc:
            logger.error("DDInter result page error: %s", exc)
            return {}

        soup = BeautifulSoup(r.text, "html.parser")
        result = {
            "url": url,
            "summary": {"Major": 0, "Moderate": 0, "Minor": 0, "Unknown": 0},
            "drug_drug_interactions": [],
            "drug_disease_interactions": [],
            "drug_food_interactions": [],
        }

        for level in ["Major", "Moderate", "Minor", "Unknown"]:
            m = re.search(rf"{level}\((\d+)\)", soup.get_text())
            if m:
                result["summary"][level] = int(m.group(1))

        for panel in soup.find_all("div", class_=lambda c: c and "layui-panel" in c and "mt-4" in c):
            badge    = panel.find("span", class_="badge")
            severity = badge.get_text(strip=True) if badge else "Unknown"

            name_div = panel.find("div", class_="drug-inter-name")
            drug_a, drug_b = "", ""
            if name_div:
                for tag in name_div.find_all("i"):
                    tag.decompose()
                parts = [t.strip() for t in name_div.get_text(separator="|").split("|") if t.strip()]
                if len(parts) >= 2:
                    drug_a, drug_b = parts[0], parts[1]
                elif parts:
                    drug_a = parts[0]

            detail_link = panel.find("a", href=re.compile(r"/server/interact/"))
            detail_url  = (BASE + detail_link["href"]) if detail_link else ""

            interaction_text, management_text = "", ""
            item_div = panel.find("div", class_="result-interaction-item")
            if item_div:
                for s in item_div.find_all("strong"):
                    s.decompose()
                interaction_text = item_div.get_text(strip=True)

            mgmt_div = panel.find("div", class_="result-management-item")
            if mgmt_div:
                for s in mgmt_div.find_all("strong"):
                    s.decompose()
                management_text = mgmt_div.get_text(strip=True)

            result["drug_drug_interactions"].append({
                "severity":    severity,
                "drug_a":      drug_a,
                "drug_b":      drug_b,
                "interaction": interaction_text,
                "management":  management_text,
                "detail_url":  detail_url,
            })

        csrf     = self._extract_csrf(soup)
        ids_slug = url.rstrip("/").split("/checker/result/")[-1]
        result["drug_disease_interactions"] = self._fetch_table(
            f"{BASE}/server/interact-with-dis/{ids_slug}/",  n_cols=6, csrf=csrf)
        result["drug_food_interactions"]    = self._fetch_table(
            f"{BASE}/server/interact-with-food/{ids_slug}/", n_cols=8, csrf=csrf)

        return result

    def _extract_csrf(self, soup) -> str:
        token = self.session.cookies.get("csrftoken", "")
        if token:
            return token
        inp = soup.find("input", {"name": "csrfmiddlewaretoken"})
        if inp:
            return inp.get("value", "")
        m = re.search(r"csrfmiddlewaretoken['\"]?\s*[:=]\s*['\"]([A-Za-z0-9]+)['\"]", soup.get_text())
        return m.group(1) if m else ""

    def _fetch_table(self, url: str, n_cols: int, csrf: str) -> list[dict]:
        try:
            payload = {
                "csrfmiddlewaretoken": csrf,
                "draw": "1", "start": "0", "length": "10000",
                "search[value]": "", "search[regex]": "false", "severity": "",
            }
            for i in range(n_cols):
                payload.update({
                    f"columns[{i}][data]": "null", f"columns[{i}][name]": "",
                    f"columns[{i}][searchable]": "true", f"columns[{i}][orderable]": "false",
                    f"columns[{i}][search][value]": "", f"columns[{i}][search][regex]": "false",
                })
            r = self.session.post(
                url, data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
                timeout=15,
            )
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as exc:
            logger.warning("DDInter table fetch failed for %s: %s", url, exc)
            return []

    @staticmethod
    def _to_chroma_documents(raw: dict, session_id: str) -> list[dict]:
        if not raw:
            return []

        documents = []

        for ddi in raw.get("drug_drug_interactions", []):
            pair_key = "-".join(sorted([ddi["drug_a"].lower(), ddi["drug_b"].lower()]))
            doc_id   = "ddi-" + hashlib.md5(pair_key.encode()).hexdigest()[:12]
            text = (
                f"Drug interaction [{ddi['severity']}]: "
                f"{ddi['drug_a']} + {ddi['drug_b']}. "
                f"{ddi['interaction']} "
                f"Management: {ddi['management']}"
            ).strip()
            documents.append({
                "id": doc_id,
                "text": text,
                "metadata": {
                    "drug_name":        ddi["drug_a"],
                    "interacting_drug": ddi["drug_b"],
                    "severity_label":   ddi["severity"],
                    "severity_score":   SEVERITY_MAP.get(ddi["severity"], 0),
                    "section_type":     "drug_drug_interaction",
                    "source":           "ddinter2_web",
                    "session_id":       session_id,
                    "detail_url":       ddi.get("detail_url", ""),
                },
            })

        for ddi in raw.get("drug_disease_interactions", []):
            doc_id = f"ddi-dis-{ddi.get('interaction_id', hashlib.md5(str(ddi).encode()).hexdigest()[:8])}"
            text = (
                f"Drug-disease interaction: {ddi.get('drugName', '')} "
                f"and {ddi.get('diseaseName', '')}. "
                f"{ddi.get('text', '')}"
            ).strip()
            documents.append({
                "id": doc_id,
                "text": text,
                "metadata": {
                    "drug_name":    ddi.get("drugName", ""),
                    "disease":      ddi.get("diseaseName", ""),
                    "level":        str(ddi.get("level", "")),
                    "section_type": "drug_disease_interaction",
                    "source":       "ddinter2_web",
                    "session_id":   session_id,
                },
            })

        summary = raw.get("summary", {})
        documents.append({
            "id": f"ddi-summary-{session_id}",
            "text": (
                f"Interaction summary: "
                f"{summary.get('Major', 0)} major, "
                f"{summary.get('Moderate', 0)} moderate, "
                f"{summary.get('Minor', 0)} minor interactions."
            ),
            "metadata": {
                "section_type":   "interaction_summary",
                "source":         "ddinter2_web",
                "session_id":     session_id,
                "major_count":    summary.get("Major", 0),
                "moderate_count": summary.get("Moderate", 0),
                "minor_count":    summary.get("Minor", 0),
            },
        })

        return documents