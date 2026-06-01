import os
import re
import sys
import tempfile
from pathlib import Path

import PIL.Image
import streamlit as st
from dotenv import load_dotenv

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

load_dotenv()

# Merge Streamlit Cloud secrets into os.environ (no-op when secrets.toml is absent).
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

# Common OTC brand -> generic, so questions like "take with Advil" resolve.
_BRAND_TO_GENERIC = {
    "advil": "ibuprofen", "motrin": "ibuprofen",
    "tylenol": "acetaminophen", "aleve": "naproxen",
    "aspirin": "aspirin", "bayer": "aspirin",
    "coumadin": "warfarin", "glucophage": "metformin",
    "lipitor": "atorvastatin", "zoloft": "sertraline",
    "prilosec": "omeprazole", "advil pm": "ibuprofen",
    # OTC cold/flu brands (kept as query tokens when no generic is enough)
    "nyquil": "nyquil", "nyquil cold and flu": "nyquil",
    "dayquil": "dayquil", "robitussin": "dextromethorphan",
    "mucinex": "guaifenesin", "benadryl": "diphenhydramine",
    "claritin": "loratadine", "zyrtec": "cetirizine",
}

# Drugs we know about (used to spot a second drug in interaction questions).
_KNOWN_DRUGS = [
    "warfarin", "metformin", "lisinopril", "atorvastatin", "amoxicillin",
    "metoprolol", "omeprazole", "amlodipine", "sertraline", "levothyroxine",
    "clopidogrel", "furosemide", "gabapentin", "hydrochlorothiazide", "fluoxetine",
    "alprazolam", "prednisone", "ciprofloxacin", "azithromycin", "losartan",
    "ibuprofen", "naproxen", "aspirin", "acetaminophen", "simvastatin",
    "tramadol", "duloxetine", "citalopram", "escitalopram",
]

_INTERACTION_WORDS = (
    "interact", "interaction", "taken with", "mix", "combine",
    "combined", "together", "can i take", "can you take", "is it safe",
    "safe to take", "along with", "alongside",
)
# "with food/water" is dosage guidance, not a drug-drug interaction.
_NON_DRUG_WITH_PHRASES = (
    "with food", "with meals", "with a meal", "with water", "with milk",
    "with or without food", "without food", "empty stomach", "with snack",
)
# Catches "take advil with tylenol" where a drug sits between take and with.
_INTERACTION_PATTERNS = (
    re.compile(r"\btake\b.+\bwith\b", re.I),
    re.compile(r"\b(is it )?safe\b.+\bwith\b", re.I),
)
_ADVERSE_WORDS = (
    "side effect", "side-effect", "adverse", "reaction", "risk", "danger",
    "harm", "safe", "warning", "overdose", "too much", "symptom",
)


def _clean_scanned_drug(name: str) -> str:
    """OCR drug name may be a brand; normalize to a generic we can query."""
    if not name:
        return ""
    low = name.strip().lower()
    if low in _BRAND_TO_GENERIC:
        return _BRAND_TO_GENERIC[low]
    for brand, generic in sorted(
        _BRAND_TO_GENERIC.items(), key=lambda x: -len(x[0])
    ):
        if brand in low:
            return generic
    return low


def _extract_mentioned_drugs(question: str) -> list[str]:
    """All drugs/brands mentioned in the question, in order of appearance."""
    q = question.lower()
    found: list[tuple[int, str]] = []
    seen: set[str] = set()

    for brand, generic in sorted(_BRAND_TO_GENERIC.items(), key=lambda x: -len(x[0])):
        idx = q.find(brand)
        if idx >= 0 and generic not in seen:
            found.append((idx, generic))
            seen.add(generic)

    for drug in _KNOWN_DRUGS:
        idx = q.find(drug)
        if idx >= 0 and drug not in seen:
            found.append((idx, drug))
            seen.add(drug)

    found.sort(key=lambda x: x[0])
    return [generic for _, generic in found]


def _is_interaction_question(q: str, scanned_drug: str = "") -> bool:
    q_lower = q.lower()

    # "Should I take this with food?" → vector_search, not interaction_check.
    if any(phrase in q_lower for phrase in _NON_DRUG_WITH_PHRASES):
        scanned = _clean_scanned_drug(scanned_drug)
        others = [d for d in _extract_mentioned_drugs(q) if d != scanned]
        if not others:
            return False

    if any(w in q_lower for w in _INTERACTION_WORDS):
        return True
    if any(p.search(q) for p in _INTERACTION_PATTERNS):
        # "take X with food" matched the pattern but is not a DDI question.
        if any(phrase in q_lower for phrase in _NON_DRUG_WITH_PHRASES):
            return False
        return True
    return False


def _resolve_query_drug(question: str, scanned_drug: str) -> str:
    """
    Drug to filter retrieval on.

    Prefer a drug named in the question over an old scan, so asking about
    warfarin while Advil is still in session targets warfarin chunks.
    """
    scanned   = _clean_scanned_drug(scanned_drug)
    mentioned = _extract_mentioned_drugs(question)
    q         = question.lower()
    refers_to_scan = scanned and any(
        tok in q for tok in ("this", "it", "these", "that")
    )

    if mentioned:
        if refers_to_scan and scanned:
            return scanned
        return mentioned[0]
    return scanned


def _resolve_interaction_pair(
    question: str, scanned_drug: str
) -> tuple[str, str]:
    """
    Pick (drug_a, drug_b) for interaction_check.

    Handles:
      - scanned label + "can I take this with tylenol?"
      - no scan + "can I take advil with tylenol?"
    Returns ("", "") when we cannot identify at least one drug.
    """
    scanned   = _clean_scanned_drug(scanned_drug)
    mentioned = _extract_mentioned_drugs(question)
    q         = question.lower()
    refers_to_scan = scanned and any(
        tok in q for tok in ("this", "it", "these", "that")
    )

    if scanned and refers_to_scan:
        others = [d for d in mentioned if d != scanned]
        if others:
            return scanned, others[0]
        return scanned, ""

    if len(mentioned) >= 2:
        return mentioned[0], mentioned[1]

    if scanned and mentioned:
        other = mentioned[0]
        if other != scanned:
            return scanned, other
        return scanned, ""

    if scanned:
        return scanned, ""

    if len(mentioned) == 1:
        return mentioned[0], ""

    return "", ""


def _find_other_drug(question: str, scanned: str) -> str:
    """Look for a second drug mentioned in the question."""
    mentioned = _extract_mentioned_drugs(question)
    for drug in mentioned:
        if drug != scanned:
            return drug
    return ""


_DIRECT_SYSTEM = """
You are a careful medical assistant for patients and clinicians.
Answer questions about medicines, dosages, side effects, and drug interactions
clearly and concisely in plain English.
Always recommend verifying with a pharmacist or doctor for personal medical decisions.
Keep answers under 200 words unless the question needs more detail.
""".strip()


def _grok_direct(question: str, scanned_drug: str) -> dict:
    """Answer using Grok directly (no RAG) when the vector DB is unavailable."""
    xai_key = os.getenv("XAI_API_KEY", "")
    if not xai_key:
        return {
            "kind": "no-key",
            "summary": (
                "XAI_API_KEY is not configured. "
                "Add it to your secrets to enable chat answers."
            ),
            "results": [],
        }
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=xai_key,
            base_url="https://api.x.ai/v1",
        )
        model = os.getenv("XAI_TEXT_MODEL", "grok-3")
        drug_line = (
            f"The user has scanned a medicine label for: {scanned_drug}.\n"
            if scanned_drug else ""
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _DIRECT_SYSTEM},
                {"role": "user", "content": f"{drug_line}Question: {question}"},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        answer = (resp.choices[0].message.content or "").strip()
        return {"kind": "grok-direct", "summary": answer, "results": []}
    except Exception as exc:
        return {
            "kind": "error",
            "summary": f"Could not reach Grok: {exc}",
            "results": [],
        }


def _db_available() -> bool:
    try:
        from knowledge.chroma_config import get_chroma_db_path
        p = get_chroma_db_path()
        return os.path.isdir(p) and any(os.scandir(p))
    except Exception:
        return False


def answer_question(question: str, scanned_drug: str) -> dict:
    """
    Route a user question to the right trained function and return
    {'kind', 'summary', 'results'} where results is a list of ranked chunks.
    """
    if not _db_available():
        return _grok_direct(question, scanned_drug)

    from knowledge.query_functions import (
        vector_search, adverse_events, interaction_check,
    )

    query_drug = _resolve_query_drug(question, scanned_drug)
    q    = question.lower()

    # 1. interaction question
    if _is_interaction_question(q, scanned_drug):
        drug_a, drug_b = _resolve_interaction_pair(question, scanned_drug)
        if drug_a and drug_b:
            results = interaction_check(drug_a, drug_b, top_n=5)
            kind    = f"interaction_check({drug_a}, {drug_b})"
        elif drug_a:
            results = interaction_check(drug_a, top_n=5)
            kind    = f"interaction_check({drug_a})"
        else:
            results = vector_search(question, top_n=5)
            kind    = "vector_search (interaction question, no drugs identified)"
        if not results and drug_a:
            fallback_q = f"{question} {drug_a}"
            if drug_b:
                fallback_q = f"{fallback_q} {drug_b}"
            results = vector_search(
                fallback_q, drug_name=drug_a, top_n=5
            )
            kind = f"vector_search(fallback, {drug_a}" + (
                f", {drug_b})" if drug_b else ")"
            )
    # 2. adverse-event / safety question
    elif any(w in q for w in _ADVERSE_WORDS):
        if query_drug:
            results = adverse_events(query_drug, reaction_hint=question, top_n=5)
            kind    = f"adverse_events({query_drug})"
        else:
            results = vector_search(question, top_n=5)
            kind    = "vector_search (no scanned drug)"
    # 3. general semantic search
    else:
        search_q = question
        if any(phrase in q for phrase in _NON_DRUG_WITH_PHRASES):
            search_q = (
                f"{question} dosage administration clinical pharmacology "
                "take with food meals absorption"
            )
        elif any(w in q for w in ("indicated", "indication", "used for", "what is it for")):
            search_q = f"{question} indications and usage"
        results = vector_search(
            search_q, drug_name=query_drug or None, top_n=5
        )
        kind = f"vector_search(drug={query_drug or 'any'})"

    if kind.startswith("interaction_check("):
        drug_context = kind[len("interaction_check(") : -1]
    elif query_drug:
        drug_context = query_drug
    else:
        drug_context = "unspecified"

    try:
        from knowledge.grok_answer import synthesize_answer
        summary = synthesize_answer(
            question,
            results,
            drug_context=drug_context,
            route_label=kind,
        )
    except Exception as exc:  # noqa: BLE001
        if results:
            summary = results[0]["text"][:400].strip()
            summary += f"\n\n_(Grok summary unavailable: {exc})_"
        else:
            summary = (
                "I couldn't find relevant information in the label database "
                f"for that question. ({exc})"
            )

    return {"kind": kind, "summary": summary, "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MedLabel", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Design system — MedTech Noir: Syne + DM Sans, deep navy, electric teal
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --bg:           #080D18;
  --bg-2:         #0D1425;
  --glass:        rgba(255,255,255,0.04);
  --glass-hover:  rgba(255,255,255,0.07);
  --border:       rgba(255,255,255,0.08);
  --accent:       #00D4B8;
  --accent-dim:   #00B09E;
  --accent-glow:  rgba(0,212,184,0.22);
  --indigo:       #6366F1;
  --text-1:       #E8EEFF;
  --text-2:       #8892A4;
  --text-3:       #4B5A6E;
  scroll-behavior: smooth;
}

*, *::before, *::after { box-sizing: border-box; }

body, .stApp, .main, [data-testid="stAppViewContainer"] {
  font-family: 'DM Sans', -apple-system, sans-serif !important;
  background: var(--bg) !important;
  color: var(--text-1) !important;
}

#MainMenu, header[data-testid="stHeader"], footer,
.stDeployButton, [data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }

.block-container,
[data-testid="stMainBlockContainer"] {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── Navbar ── */
.ml-nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 80px; height: 72px;
  background: rgba(8,13,24,0.85);
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  position: sticky; top: 0; z-index: 1000;
}
.ml-logo {
  display: flex; align-items: center; gap: 10px;
  font-family: 'Syne', sans-serif; font-weight: 800;
  font-size: 18px; color: var(--text-1); text-decoration: none;
}
.ml-logo-icon {
  width: 34px; height: 34px;
  background: linear-gradient(135deg, #00D4B8 0%, #00A896 100%);
  border-radius: 9px; flex-shrink: 0;
  box-shadow: 0 0 16px rgba(0,212,184,0.35);
}
.ml-nav-links { display: flex; align-items: center; gap: 32px; }
.ml-nav-links a {
  color: var(--text-2); text-decoration: none;
  font-size: 14px; font-weight: 500;
  transition: color .2s;
}
.ml-nav-links a:hover { color: var(--accent); }

/* ── Hero ── */
.ml-hero-wrap {
  position: relative; overflow: hidden;
  background: var(--bg);
  padding: 110px 20px 100px;
  text-align: center;
}
.ml-hero-wrap::before {
  content: '';
  position: absolute; inset: 0;
  background:
    radial-gradient(ellipse 700px 500px at 20% 50%, rgba(0,212,184,0.09) 0%, transparent 70%),
    radial-gradient(ellipse 600px 400px at 80% 60%, rgba(99,102,241,0.07) 0%, transparent 70%);
  animation: orb-drift 14s ease-in-out infinite alternate;
  pointer-events: none;
}
@keyframes orb-drift {
  from { transform: scale(1) translateY(0px); }
  to   { transform: scale(1.07) translateY(-18px); }
}
.ml-hero-grid {
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(0,212,184,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,184,0.025) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none; z-index: 0;
}
.ml-hero-inner { position: relative; z-index: 1; }
.ml-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(0,212,184,0.09); border: 1px solid rgba(0,212,184,0.22);
  border-radius: 999px; padding: 6px 16px;
  font-size: 12px; font-weight: 600; letter-spacing: .8px;
  color: var(--accent); text-transform: uppercase; margin-bottom: 32px;
}
.ml-badge-dot {
  width: 7px; height: 7px; background: var(--accent);
  border-radius: 50%; display: inline-block;
  animation: pulse-dot 2.2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.45; transform: scale(0.65); }
}
.ml-hero h1 {
  font-family: 'Syne', sans-serif;
  font-size: clamp(38px, 5.5vw, 68px); font-weight: 800;
  color: var(--text-1); line-height: 1.07;
  max-width: 860px; margin: 0 auto 24px;
  letter-spacing: -.5px;
}
.ml-hero h1 .hl {
  background: linear-gradient(90deg, #00D4B8 0%, #6366F1 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.ml-hero-sub {
  font-size: 18px; color: var(--text-2); max-width: 520px;
  margin: 0 auto 52px; line-height: 1.7;
}

/* see-how anchor button */
.see-how-btn {
  display: flex; align-items: center; justify-content: center;
  width: 100%; padding: 12px 20px; border-radius: 12px;
  background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
  color: #E8EEFF !important; font-family: 'DM Sans', sans-serif;
  font-size: 15px; font-weight: 600; text-decoration: none !important;
  transition: border-color .2s, color .2s, background .2s;
  cursor: pointer;
}
.see-how-btn:hover {
  border-color: rgba(0,212,184,0.5); color: #00D4B8 !important;
  background: rgba(0,212,184,0.06);
}

/* ── Trust strip ── */
.ml-trust-strip {
  display: flex; justify-content: center; align-items: center;
  flex-wrap: wrap; gap: 32px;
  padding: 20px 80px;
  border-top: 1px solid rgba(255,255,255,0.06);
  border-bottom: 1px solid rgba(255,255,255,0.06);
  background: rgba(255,255,255,0.015);
}
.ml-trust-item {
  display: flex; align-items: center; gap: 7px;
  font-size: 13px; color: var(--text-3); font-weight: 500;
}

/* ── Features ── */
.ml-features { padding: 96px 80px; background: var(--bg); }
.ml-features-eyebrow {
  text-align: center; font-size: 12px; font-weight: 600;
  letter-spacing: 1.2px; text-transform: uppercase;
  color: var(--accent); margin-bottom: 14px;
}
.ml-features-title {
  text-align: center;
  font-family: 'Syne', sans-serif;
  font-size: clamp(28px, 3.5vw, 42px); font-weight: 800;
  color: var(--text-1); margin-bottom: 64px; letter-spacing: -.3px;
}
.ml-cards {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 24px; max-width: 1100px; margin: 0 auto;
}
@media (max-width: 900px) { .ml-cards { grid-template-columns: 1fr; } }
.ml-card {
  background: rgba(255,255,255,0.04);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px; padding: 36px 32px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
  transition: transform .25s, box-shadow .25s, border-color .25s, background .25s;
}
.ml-card:hover {
  transform: translateY(-5px);
  border-color: rgba(0,212,184,0.28);
  background: rgba(255,255,255,0.07);
  box-shadow: 0 12px 48px rgba(0,212,184,0.07);
}
.ml-card-icon {
  width: 56px; height: 56px;
  background: rgba(0,212,184,0.1); border-radius: 16px;
  margin-bottom: 24px;
  display: flex; align-items: center; justify-content: center;
}
.ml-card-icon svg { width: 26px; height: 26px; }
.ml-card h3 {
  font-family: 'Syne', sans-serif;
  font-size: 18px; font-weight: 700; color: #E8EEFF; margin-bottom: 12px;
}
.ml-card p { font-size: 14px; color: #8892A4; line-height: 1.7; }

/* ── How it works ── */
.ml-how {
  padding: 96px 80px;
  background: linear-gradient(180deg, #080D18 0%, #0A1120 100%);
  border-top: 1px solid rgba(255,255,255,0.06);
}
.ml-how-eyebrow {
  text-align: center; font-size: 12px; font-weight: 600;
  letter-spacing: 1.2px; text-transform: uppercase;
  color: #6366F1; margin-bottom: 14px;
}
.ml-how-title {
  text-align: center;
  font-family: 'Syne', sans-serif;
  font-size: clamp(28px, 3.5vw, 42px); font-weight: 800;
  color: #E8EEFF; margin-bottom: 64px; letter-spacing: -.3px;
}
.ml-steps {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 0; max-width: 860px; margin: 0 auto; position: relative;
}
.ml-steps::before {
  content: ''; position: absolute;
  top: 27px; left: calc(16.66% + 28px); right: calc(16.66% + 28px);
  height: 1px;
  background: linear-gradient(90deg, rgba(0,212,184,0.4) 0%, rgba(99,102,241,0.4) 100%);
}
.ml-step {
  display: flex; flex-direction: column; align-items: center;
  text-align: center; padding: 0 28px;
}
.ml-step-num {
  width: 54px; height: 54px; border-radius: 50%;
  background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10);
  display: flex; align-items: center; justify-content: center;
  font-family: 'Syne', sans-serif; font-size: 17px; font-weight: 800;
  color: #00D4B8; margin-bottom: 22px;
  position: relative; z-index: 1;
}
.ml-step h4 {
  font-family: 'Syne', sans-serif; font-size: 16px; font-weight: 700;
  color: #E8EEFF; margin-bottom: 8px;
}
.ml-step p { font-size: 13px; color: #8892A4; line-height: 1.65; }

/* ── Bottom CTA section ── */
.ml-cta-section {
  text-align: center; padding: 88px 20px;
  background: var(--bg);
  border-top: 1px solid rgba(255,255,255,0.06);
}
.ml-cta-section h2 {
  font-family: 'Syne', sans-serif;
  font-size: clamp(26px, 3.5vw, 40px); font-weight: 800;
  color: #E8EEFF; margin-bottom: 12px; letter-spacing: -.3px;
}
.ml-cta-section p {
  color: #8892A4; font-size: 16px; margin-bottom: 40px;
}

/* ── Footer ── */
.ml-footer {
  padding: 40px 80px; border-top: 1px solid rgba(255,255,255,0.06);
  text-align: center; color: #4B5A6E; font-size: 13px;
  background: #080D18;
}
.ml-footer a { color: #8892A4 !important; text-decoration: none; }
.ml-footer a:hover { color: #00D4B8 !important; }

/* ── Scanner page ── */
.ml-scanner { max-width: 1320px; margin: 0 auto; padding: 40px 56px; }
.ml-scanner-header { margin-bottom: 36px; }
.ml-scanner-title {
  font-family: 'Syne', sans-serif;
  font-size: 28px; font-weight: 800; color: #E8EEFF; margin-bottom: 6px;
}
.ml-scanner-desc { font-size: 15px; color: #8892A4; }
.ml-label {
  font-size: 11px; font-weight: 600; color: #4B5A6E;
  letter-spacing: .9px; text-transform: uppercase; margin-bottom: 10px;
}

/* ── Streamlit widget overrides ── */
.stButton > button {
  background: linear-gradient(135deg, #00D4B8 0%, #00A896 100%) !important;
  color: #080D18 !important;
  border: none !important; border-radius: 12px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important; font-size: 15px !important;
  transition: all .2s ease !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 28px rgba(0,212,184,0.28) !important;
}
.stButton > button[kind="secondary"] {
  background: rgba(255,255,255,0.04) !important;
  color: #E8EEFF !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover {
  border-color: rgba(0,212,184,0.4) !important;
  color: #00D4B8 !important;
  background: rgba(0,212,184,0.05) !important;
  transform: none !important;
}
.back-btn button {
  background: transparent !important; color: #00D4B8 !important;
  border: none !important; box-shadow: none !important;
  font-size: 14px !important; padding: 0 !important;
  font-weight: 500 !important; transform: none !important;
}
.back-btn button:hover { opacity: 0.75 !important; }

[data-testid="stFileUploaderDropzone"] {
  border: 1.5px dashed rgba(0,212,184,0.35) !important;
  border-radius: 16px !important;
  background: rgba(0,212,184,0.025) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: #00D4B8 !important;
  background: rgba(0,212,184,0.055) !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0; background: transparent;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
.stTabs [data-baseweb="tab"] {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important; color: #8892A4 !important;
  background: transparent; border: none; padding: 10px 22px;
}
.stTabs [aria-selected="true"] {
  color: #00D4B8 !important;
  border-bottom: 2px solid #00D4B8 !important;
}

[data-testid="stMetric"] {
  background: rgba(255,255,255,0.04) !important;
  backdrop-filter: blur(12px);
  border-radius: 14px; padding: 18px 22px;
  border: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stMetric"] label {
  color: #8892A4 !important; font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: #E8EEFF !important; font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
}

.stTextArea textarea,
[data-testid="stChatInput"] textarea {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 14px !important;
  color: #E8EEFF !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stTextArea textarea:focus,
[data-testid="stChatInput"] textarea:focus {
  border-color: #00D4B8 !important;
  box-shadow: 0 0 0 2px rgba(0,212,184,0.15) !important;
}

.stRadio label, .stRadio p {
  font-family: 'DM Sans', sans-serif !important;
  color: #8892A4 !important;
}
div[data-baseweb="radio"] > label > div:first-child > div {
  border-color: #00D4B8 !important;
}
div[data-baseweb="radio"] > label {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 10px !important;
  padding: 10px 14px !important;
  margin-bottom: 6px !important;
  transition: border-color .2s, background .2s;
}
div[data-baseweb="radio"] > label:hover {
  border-color: rgba(0,212,184,0.3) !important;
  background: rgba(0,212,184,0.04) !important;
}

.stAlert, .element-container .stSuccess,
.element-container .stInfo,
.element-container .stWarning,
.element-container .stError {
  border-radius: 14px !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stSpinner p {
  font-family: 'DM Sans', sans-serif !important;
  color: #8892A4 !important;
}
.stExpander {
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 14px !important;
  background: rgba(255,255,255,0.03) !important;
}
[data-testid="stChatMessage"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 14px !important;
}
[data-testid="stImage"] img {
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.08);
}
.stCaption p, small {
  color: #4B5A6E !important;
  font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stDivider"] { border-color: rgba(255,255,255,0.07) !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080D18; }
::-webkit-scrollbar-thumb { background: rgba(0,212,184,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,184,0.45); }

/* ── Dark theme: patch all remaining Streamlit light-bg leaks ── */

/* Root containers */
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.stApp, .main {
  background-color: #080D18 !important;
}

/* Block container — Streamlit adds its own padding/bg */
[data-testid="stMainBlockContainer"],
[data-testid="block-container"],
.block-container {
  background: transparent !important;
  padding: 0 !important;
  max-width: 100% !important;
}

/* Every st.write / st.markdown paragraph */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] code,
.stMarkdown p, .stMarkdown li {
  color: #E8EEFF !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* st.write plain text */
[data-testid="stText"] {
  color: #E8EEFF !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* Widget labels */
label, .stSelectbox label, .stRadio label,
[data-testid="stWidgetLabel"] p,
[data-baseweb="label"] {
  color: #8892A4 !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* st.info / st.success / st.warning / st.error */
[data-testid="stNotification"],
[data-testid="stAlert"] {
  background: #0D1425 !important;
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  color: #E8EEFF !important;
}
[data-testid="stAlert"] p { color: #E8EEFF !important; }

/* st.caption */
[data-testid="stCaptionContainer"] p,
.stCaption, .stCaption p {
  color: #4B5A6E !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* Text area */
.stTextArea label { color: #8892A4 !important; }
.stTextArea textarea {
  background: #0D1425 !important;
  color: #E8EEFF !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
}

/* File uploader label text */
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploadDropzone"] p {
  color: #8892A4 !important;
}

/* Expander header */
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderDetails"],
details summary p,
.stExpander details summary span {
  color: #E8EEFF !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-panel"] {
  background: transparent !important;
  padding-top: 20px;
}

/* Chat input area */
[data-testid="stChatInputContainer"] {
  background: #080D18 !important;
  border-top: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stChatInput"] {
  background: #0D1425 !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 14px !important;
}

/* Chat message avatars and text */
[data-testid="stChatMessage"] p { color: #E8EEFF !important; }

/* Metric delta */
[data-testid="stMetricDelta"] svg { stroke: #00D4B8 !important; }

/* st.divider */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* Spinner text */
[data-testid="stSpinner"] p { color: #8892A4 !important; }

/* Selectbox / dropdown */
[data-baseweb="select"] > div {
  background: #0D1425 !important;
  border-color: rgba(255,255,255,0.08) !important;
  color: #E8EEFF !important;
}
[data-baseweb="popover"] ul {
  background: #0D1425 !important;
  color: #E8EEFF !important;
}

/* Code blocks */
code, pre {
  background: #0D1425 !important;
  color: #00D4B8 !important;
  border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Navigation state
# ─────────────────────────────────────────────────────────────────────────────
_show_scanner = st.session_state.get("show_scanner", False)

# ─────────────────────────────────────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────────
if not _show_scanner:
    st.markdown("""
<nav class="ml-nav">
  <a class="ml-logo" href="#">
    <div class="ml-logo-icon"></div>
    MedLabel
  </a>
  <div class="ml-nav-links">
    <a href="#ml-features">Features</a>
    <a href="#ml-how">How it works</a>
    <a href="#ml-safety">Safety</a>
  </div>
</nav>""", unsafe_allow_html=True)

    st.markdown("""
<div class="ml-hero-wrap">
  <div class="ml-hero-grid"></div>
  <div class="ml-hero-inner">
    <div class="ml-badge"><span class="ml-badge-dot"></span>FDA-Grounded &nbsp;·&nbsp; Free &nbsp;·&nbsp; Private</div>
    <div class="ml-hero">
      <h1>Read any medicine label<br><span class="hl">in plain English</span></h1>
      <p class="ml-hero-sub">
        Point your phone at any pill bottle, box, or blister pack. MedLabel reads it,
        explains it clearly, and answers your questions — grounded in official FDA data.
      </p>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    _, c1, c2, _ = st.columns([2, 1.2, 1.2, 2])
    with c1:
        if st.button("Scan a label →", key="hero_cta", type="primary", use_container_width=True):
            st.session_state["show_scanner"] = True
            st.rerun()
    with c2:
        st.markdown("""<a href="#ml-features" class="see-how-btn">See how it works</a>""",
                    unsafe_allow_html=True)

    st.markdown("""
<div class="ml-trust-strip">
  <div class="ml-trust-item">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#00D4B8" stroke-width="2.5" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg>
    5,976 FDA label chunks
  </div>
  <div class="ml-trust-item">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#00D4B8" stroke-width="2.5" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg>
    YOLO + OCR pipeline
  </div>
  <div class="ml-trust-item">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#00D4B8" stroke-width="2.5" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg>
    Grok &amp; Gemini AI
  </div>
  <div class="ml-trust-item">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#00D4B8" stroke-width="2.5" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg>
    No data stored
  </div>
</div>

<section class="ml-features" id="ml-features">
  <div class="ml-features-eyebrow">Capabilities</div>
  <div class="ml-features-title">Built for trust and clarity</div>
  <div class="ml-cards">
    <div class="ml-card">
      <div class="ml-card-icon">
        <svg width="28" height="28" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M6 14V8a2 2 0 0 1 2-2h6" stroke="#00D4B8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M34 14V8a2 2 0 0 0-2-2h-6" stroke="#00D4B8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M6 26v6a2 2 0 0 0 2 2h6" stroke="#00D4B8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M34 26v6a2 2 0 0 1-2 2h-6" stroke="#00D4B8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          <line x1="6" y1="20" x2="34" y2="20" stroke="#00D4B8" stroke-width="1.8" stroke-linecap="round" stroke-dasharray="3 2"/>
          <circle cx="20" cy="20" r="2.5" fill="#00D4B8"/>
        </svg>
      </div>
      <h3>Smart scanning</h3>
      <p>YOLO geometry detection routes each photo to the right pipeline — PaddleOCR for flat labels, Vision AI for curved bottles.</p>
    </div>
    <div class="ml-card">
      <div class="ml-card-icon">
        <svg width="28" height="28" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M8 10a3 3 0 0 1 3-3h18a3 3 0 0 1 3 3v13a3 3 0 0 1-3 3H22l-5 5v-5H11a3 3 0 0 1-3-3V10Z" stroke="#00D4B8" stroke-width="2" stroke-linejoin="round"/>
          <line x1="13" y1="15" x2="27" y2="15" stroke="#00D4B8" stroke-width="1.8" stroke-linecap="round"/>
          <line x1="13" y1="20" x2="23" y2="20" stroke="#00D4B8" stroke-width="1.8" stroke-linecap="round" opacity="0.6"/>
        </svg>
      </div>
      <h3>Plain-English answers</h3>
      <p>Medical jargon simplified to a 5th-grade reading level. Ask anything — side effects, dosage, interactions — in natural language.</p>
    </div>
    <div class="ml-card">
      <div class="ml-card-icon">
        <svg width="28" height="28" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M20 4L8 9v10c0 7.18 5.12 13.9 12 15.5C27.88 32.9 33 26.18 33 19V9L20 4Z" stroke="#00D4B8" stroke-width="2" stroke-linejoin="round"/>
          <path d="M14 20l4 4 8-8" stroke="#00D4B8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <h3>Grounded in FDA data</h3>
      <p>Every answer is retrieved from 5,976 chunks of official FDA drug labels — never hallucinated, always cited.</p>
    </div>
  </div>
</section>

<section class="ml-how" id="ml-how">
  <div class="ml-how-eyebrow">Process</div>
  <div class="ml-how-title">Three steps to clarity</div>
  <div class="ml-steps">
    <div class="ml-step">
      <div class="ml-step-num">01</div>
      <h4>Upload a photo</h4>
      <p>Take a picture of any medicine label — pill bottle, box, or blister pack.</p>
    </div>
    <div class="ml-step">
      <div class="ml-step-num">02</div>
      <h4>AI reads the label</h4>
      <p>YOLO detects geometry, then OCR or Vision AI extracts the text accurately.</p>
    </div>
    <div class="ml-step">
      <div class="ml-step-num">03</div>
      <h4>Get plain answers</h4>
      <p>Ask questions in natural language. Grok answers using official FDA data.</p>
    </div>
  </div>
</section>""", unsafe_allow_html=True)

    st.markdown("""
<div class="ml-cta-section">
  <h2>Ready to scan your first label?</h2>
  <p style="color:#8892A4;font-size:16px;margin-bottom:8px;">Takes under 10 seconds. No account needed.</p>
</div>""", unsafe_allow_html=True)

    _, cta_col, _ = st.columns([3, 2, 3])
    with cta_col:
        if st.button("Get started — scan a label", key="bottom_cta", type="primary", use_container_width=True):
            st.session_state["show_scanner"] = True
            st.rerun()

    st.markdown("""
<footer class="ml-footer" id="ml-safety">
  <p>&#169; 2026 MedLabel &nbsp;&middot;&nbsp; <a href="#">Privacy</a> &nbsp;&middot;&nbsp; <a href="#">Terms</a></p>
  <p style="margin-top:8px;font-size:12px;">An informational tool only &#8212; not a substitute for professional medical advice. Always verify with your pharmacist.</p>
</footer>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SCANNER
# ─────────────────────────────────────────────────────────────────────────────
else:
    gemini_status = "Set" if os.getenv("GEMINI_API_KEY") else "Missing"

    st.markdown("""
<nav class="ml-nav">
  <a class="ml-logo" href="#">
    <div class="ml-logo-icon"></div>
    MedLabel
  </a>
  <div class="ml-nav-links">
    <span style="font-size:13px;color:#4B5A6E;font-family:'DM Sans',sans-serif;">
      AI can make mistakes &#8212; always verify with your pharmacist.
    </span>
  </div>
</nav>""", unsafe_allow_html=True)

    st.markdown('<div class="ml-scanner">', unsafe_allow_html=True)

    _bc, _tc = st.columns([1, 7])
    with _bc:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back", key="back_home"):
            st.session_state["show_scanner"] = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="ml-scanner-header">
  <div class="ml-scanner-title">Scan your medicine label</div>
  <div class="ml-scanner-desc">Upload a photo &#8212; MedLabel reads it, simplifies it, and answers your questions.</div>
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="ml-label">Label type</div>', unsafe_allow_html=True)
        packaging_type = st.radio(
            "Label type",
            options=["cylindrical", "flat", "auto"],
            format_func=lambda x: {
                "cylindrical": "\U0001fad9  Bottle / curved label  (xAI Vision)",
                "flat":        "\U0001f4e6  Flat label  (PaddleOCR)",
                "auto":        "\U0001f916  Auto-detect  (YOLO)",
            }[x],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown('<div class="ml-label" style="margin-top:24px;">Upload label photo</div>',
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload label photo",
            type=["jpg", "jpeg", "png", "webp", "heic", "heif"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            raw = uploaded_file.getvalue()
            fname = uploaded_file.name
            img = None
            try:
                from vision.image_io import open_image

                img = open_image(data=raw, filename=fname)
            except ImportError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not open image: {exc}")

            if img is not None:
                st.image(img, caption="Uploaded image", use_container_width=True)

            if img is not None and st.button("Analyze Label", type="primary", use_container_width=True):
                tmp_path = None
                try:
                    from vision.image_io import materialize_upload, needs_heif

                    if needs_heif(fname, raw):
                        tmp_path = materialize_upload(raw, fname, for_opencv=True)
                    else:
                        suffix = Path(fname).suffix or ".png"
                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                            tmp.write(raw)
                            tmp_path = tmp.name

                    if packaging_type == "auto":
                        with st.spinner("Detecting packaging type with YOLO…"):
                            from vision.detector import YOLORouter
                            router = YOLORouter()
                            pkg, _bbox, _conf = router.detect_geometry(tmp_path)
                            note = "" if router._trained else " *(fallback — model not trained)*"
                            st.info(f"YOLO detected: **{'Flat' if pkg == 'flat' else 'Cylindrical'}**{note}")
                    else:
                        pkg = packaging_type

                    path_label = (
                        "Reading bottle label with xAI Vision…"
                        if pkg == "cylindrical"
                        else "Reading flat label with PaddleOCR…"
                    )
                    with st.spinner(path_label):
                        from vision.ocr import run_ocr
                        result = run_ocr(tmp_path, packaging_type=pkg)

                    if "reupload_required" in result.hallucination_flags:
                        conf_pct = int(result.confidence * 100)
                        st.warning(
                            f"Image quality too low (confidence: {conf_pct}%, threshold: 75%). "
                            "Please retake the photo with better lighting and try again."
                        )
                    else:
                        st.session_state["ocr_result"]         = result
                        st.session_state["simplified_summary"] = None
                        st.session_state["chat_history"]       = []
                        st.success(f"Done — confidence: {int(result.confidence * 100)}%")

                except ImportError as exc:
                    st.error(f"PaddleOCR is not installed: {exc}")
                except Exception as exc:
                    st.error(f"OCR failed: {exc}")
                finally:
                    if tmp_path:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

    with col2:
        tab1, tab2, tab3 = st.tabs(["Extracted Info", "Simplified Info", "Chatbot"])

        with tab1:
            result = st.session_state.get("ocr_result")
            if result:
                c1, c2 = st.columns(2)
                c1.metric("Drug Name", result.drug_name or "—")
                c2.metric("Dosage", result.dosage or "—")
                st.divider()
                st.caption("All text detected on label:")
                st.text_area("Raw label text", result.raw_text, height=280, disabled=True)
                st.caption(f"Path: {result.path_used}  ·  Confidence: {result.confidence:.0%}")
            else:
                st.info("Upload and analyze a label to see results here.")

        with tab2:
            result = st.session_state.get("ocr_result")
            if result:
                cached = st.session_state.get("simplified_summary")
                if cached:
                    st.markdown(cached)
                    if st.button("Regenerate Summary", key="regen"):
                        st.session_state["simplified_summary"] = None
                        st.rerun()
                else:
                    if gemini_status == "Missing":
                        st.warning("Gemini API key not set — add GEMINI_API_KEY to secrets.")
                    else:
                        if st.button("Generate Plain-English Summary", type="primary"):
                            with st.spinner("Asking Gemini to simplify the label…"):
                                from agent.simplifier import simplify_label
                                summary = simplify_label(result)
                                st.session_state["simplified_summary"] = summary
                            st.markdown(summary)
            else:
                st.info("Analyze a label to see a plain-English summary.")

        with tab3:
            result = st.session_state.get("ocr_result")
            scanned_drug = (result.drug_name if result else "") or ""

            if scanned_drug:
                st.caption(f"Answering about: **{scanned_drug}**")
            else:
                st.caption("Tip: scan a label first, or ask any general drug question.")

            for msg in st.session_state.get("chat_history", []):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            user_input = st.chat_input("e.g. What are the side effects? Can I take this with Advil?")
            if user_input:
                st.session_state.setdefault("chat_history", []).append(
                    {"role": "user", "content": user_input}
                )
                with st.chat_message("user"):
                    st.write(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        try:
                            ans = answer_question(user_input, scanned_drug)
                        except Exception as exc:
                            ans = {"kind": "error",
                                   "summary": f"Something went wrong: {exc}",
                                   "results": []}

                    st.write(ans["summary"])
                    st.caption(f"Route: `{ans['kind']}`")

                    if ans["results"]:
                        with st.expander(f"{len(ans['results'])} FDA label sources"):
                            for i, r in enumerate(ans["results"], 1):
                                meta = r.get("metadata", {})
                                st.markdown(
                                    f"**{i}. {meta.get('drug_name','?')}** — "
                                    f"*{meta.get('section_type','?')}* "
                                    f"(score {r.get('final_score',0):.2f})"
                                )
                                st.write(r["text"][:500] + "…")
                                st.divider()

                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": ans["summary"]}
                )

    st.markdown('</div>', unsafe_allow_html=True)
