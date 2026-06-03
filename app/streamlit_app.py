import os
import re
import sys
import tempfile
from pathlib import Path

import PIL.Image
import streamlit as st
import streamlit.components.v1 as components
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
    "drink alcohol", "drinking alcohol", "have a drink", "have alcohol",
)

# Substances that can interact with drugs but aren't in the drug DB.
# Recognised so they appear as drug_b in interaction_check.
_KNOWN_SUBSTANCES = [
    "alcohol", "grapefruit", "caffeine", "tobacco", "smoking",
    "milk", "antacid", "dairy",
]
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
    """All drugs/brands/substances mentioned in the question, in order of appearance."""
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

    for substance in _KNOWN_SUBSTANCES:
        idx = q.find(substance)
        if idx >= 0 and substance not in seen:
            found.append((idx, substance))
            seen.add(substance)

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
                "Model API key is not configured. "
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
        return {"kind": "our-model", "summary": answer, "results": []}
    except Exception as exc:
        return {
            "kind": "error",
            "summary": f"Could not reach our model: {exc}",
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

    try:
        from knowledge.query_functions import (
            vector_search, adverse_events, interaction_check,
        )
    except Exception:
        return _grok_direct(question, scanned_drug)

    query_drug = _resolve_query_drug(question, scanned_drug)
    q    = question.lower()

    # 1. interaction question
    if _is_interaction_question(q, scanned_drug):
        drug_a, drug_b = _resolve_interaction_pair(question, scanned_drug)
        if drug_a and drug_b:
            results = interaction_check(drug_a, drug_b, question=question, top_n=3)
            kind    = f"interaction_check({drug_a}, {drug_b})"
        elif drug_a:
            results = interaction_check(drug_a, question=question, top_n=3)
            kind    = f"interaction_check({drug_a})"
        else:
            results = vector_search(question, top_n=3)
            kind    = "vector_search (interaction question, no drugs identified)"
        if not results and drug_a:
            fallback_q = f"{question} {drug_a}"
            if drug_b:
                fallback_q = f"{fallback_q} {drug_b}"
            results = vector_search(
                fallback_q, drug_name=drug_a, top_n=3
            )
            kind = f"vector_search(fallback, {drug_a}" + (
                f", {drug_b})" if drug_b else ")"
            )
    # 2. adverse-event / safety question
    elif any(w in q for w in _ADVERSE_WORDS):
        if query_drug:
            results = adverse_events(query_drug, reaction_hint=question, top_n=3)
            kind    = f"adverse_events({query_drug})"
        else:
            results = vector_search(question, top_n=3)
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
            search_q, drug_name=query_drug or None, top_n=3
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
    except RuntimeError as exc:
        # Embedder failed (BGE-M3 not loaded) — fall back to Grok direct
        if "BGE-M3" in str(exc) or "not loaded" in str(exc):
            return _grok_direct(question, scanned_drug)
        if results:
            summary = results[0]["text"][:400].strip()
        else:
            summary = "I couldn't find relevant information for that question."
    except Exception as exc:
        if results:
            summary = results[0]["text"][:400].strip()
            summary += f"\n\n_(Summary unavailable: {exc})_"
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
# Design system — Pharma Editorial: Cormorant + Jakarta Sans, citrus accent
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,600;1,700;1,800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
  --bg:       #07080D;
  --surface:  #0E0F16;
  --surface2: #151620;
  --border:   rgba(255,255,255,0.07);
  --lime:     #C8FF57;
  --lime-dim: rgba(200,255,87,0.07);
  --lime-b:   rgba(200,255,87,0.18);
  --red:      #FF4D4D;
  --text:     #F0F4FF;
  --text2:    #8290A4;
  --text3:    #5A6A80;
  scroll-behavior: smooth;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; }

body, .stApp, .main, [data-testid="stAppViewContainer"] {
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}

#MainMenu, header[data-testid="stHeader"],
footer:not(.ml-footer),
.stDeployButton, [data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }

.block-container, [data-testid="stMainBlockContainer"] {
  padding: 0 !important; max-width: 100% !important;
}

/* ────────────────── NAVBAR ────────────────── */
.ml-nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 7vw; height: 68px;
  background: rgba(7,8,13,0.94);
  backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
  border-bottom: 1px solid var(--border);
  position: sticky; top: 0; z-index: 1000;
}
.ml-logo {
  display: flex; align-items: center; gap: 10px;
  text-decoration: none !important; flex-shrink: 0;
}
.ml-logo:hover { text-decoration: none !important; }
.ml-logo * { text-decoration: none !important; }
.ml-logo-mark {
  width: 34px; height: 34px; border-radius: 9px;
  background: var(--lime); flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 0 1px rgba(200,255,87,0.35), inset 0 1px 0 rgba(255,255,255,0.25);
}
.ml-logo-mark svg { width: 20px; height: 20px; }
.ml-logo-wordmark {
  display: flex; align-items: baseline; gap: 0;
  line-height: 1;
}
.ml-logo-med {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-style: italic; font-weight: 700;
  font-size: 20px; color: var(--text);
  letter-spacing: -.3px;
}
.ml-logo-label {
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-weight: 500; font-size: 15px;
  color: var(--text2); letter-spacing: .3px;
}
.ml-nav-links { display: flex; align-items: center; gap: 28px; }
.ml-nav-links a {
  color: var(--text2); text-decoration: none;
  font-size: 13.5px; font-weight: 500;
  letter-spacing: .1px;
  transition: color .18s;
}
.ml-nav-links a:hover { color: var(--text); }

/* ────────────────── HERO ────────────────── */
.ml-hero-wrap {
  position: relative; overflow: hidden;
  background: var(--bg);
  padding: 100px 7vw 0;
}
.ml-hero-noise {
  position: absolute; inset: 0; pointer-events: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
  background-size: 200px 200px; opacity: .5;
}
.ml-hero-glow {
  position: absolute; pointer-events: none;
  width: 600px; height: 600px; border-radius: 50%;
  background: radial-gradient(circle, rgba(200,255,87,0.06) 0%, transparent 70%);
  top: -100px; left: -100px;
  animation: glow-drift 12s ease-in-out infinite alternate;
}
@keyframes glow-drift {
  from { transform: translate(0,0) scale(1); }
  to   { transform: translate(120px, 60px) scale(1.15); }
}
.ml-hero-inner { position: relative; z-index: 1; max-width: 900px; }
.ml-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 400;
  letter-spacing: 1.5px; text-transform: uppercase;
  color: var(--lime); margin-bottom: 28px;
}
.ml-eyebrow-line {
  display: inline-block; width: 28px; height: 1px;
  background: var(--lime); opacity: .7;
}
.ml-hero h1 {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-style: italic; font-weight: 700;
  font-size: clamp(52px, 7.5vw, 100px);
  color: var(--text); line-height: 1.02;
  letter-spacing: -1px; margin-bottom: 32px;
}
.ml-hero h1 em {
  font-style: normal;
  color: var(--lime);
}
.ml-hero-sub {
  font-size: clamp(15px, 1.6vw, 18px);
  color: var(--text2); max-width: 480px;
  line-height: 1.75; margin-bottom: 52px;
  font-weight: 300;
}
.ml-hero-divider {
  width: 100%; height: 1px;
  background: linear-gradient(90deg, var(--lime-b) 0%, transparent 60%);
  margin-bottom: 0; margin-top: 56px;
}

/* ────────────────── CTA BUTTONS (host Streamlit) ────────────────── */
.ml-cta-wrap {
  display: flex; align-items: center; gap: 12px;
  margin-bottom: 0;
}
.see-how-btn {
  display: inline-flex; align-items: center; justify-content: center;
  padding: 13px 28px; border-radius: 10px;
  border: 1px solid var(--border);
  color: var(--text2) !important; font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 14px; font-weight: 500; text-decoration: none !important;
  transition: border-color .2s, color .2s;
  white-space: nowrap;
}
.see-how-btn:hover { border-color: rgba(255,255,255,0.2); color: var(--text) !important; }

/* ────────────────── STATS BAR ────────────────── */
.ml-stats {
  display: grid; grid-template-columns: repeat(4, 1fr);
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
  background: var(--surface);
}
.ml-stat {
  padding: 28px 7vw;
  border-right: 1px solid var(--border);
}
.ml-stat:last-child { border-right: none; }
.ml-stat-num {
  font-family: 'Space Mono', monospace;
  font-size: clamp(22px, 2.5vw, 30px); font-weight: 700;
  color: var(--lime); letter-spacing: -1px;
  line-height: 1.1; margin-bottom: 4px;
}
.ml-stat-label {
  font-size: 12px; color: var(--text3);
  font-weight: 500; letter-spacing: .3px;
  text-transform: uppercase;
}

/* ────────────────── FEATURES ────────────────── */
.ml-features {
  padding: 100px 7vw;
  background: var(--bg);
}
.ml-section-tag {
  font-family: 'Space Mono', monospace;
  font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--text3); margin-bottom: 16px;
}
.ml-section-tag::before { content: '// '; color: var(--lime); }
.ml-features-title {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-weight: 700; font-style: italic;
  font-size: clamp(32px, 4vw, 52px);
  color: var(--text); margin-bottom: 64px; line-height: 1.1;
  max-width: 480px;
}
.ml-cards {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 1px; background: var(--border);
  border: 1px solid var(--border); border-radius: 16px; overflow: hidden;
}
@media (max-width: 900px) { .ml-cards { grid-template-columns: 1fr; } }
.ml-card {
  background: var(--surface);
  padding: 40px 36px 44px;
  transition: background .2s;
  position: relative;
}
.ml-card::before {
  content: ''; position: absolute;
  top: 0; left: 0; right: 0; height: 2px;
  background: transparent;
  transition: background .2s;
}
.ml-card:hover { background: var(--surface2); }
.ml-card:hover::before { background: var(--lime); }
.ml-card-num {
  font-family: 'Space Mono', monospace;
  font-size: 11px; color: var(--lime); letter-spacing: 1px;
  margin-bottom: 20px; opacity: .7;
}
.ml-card h3 {
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 17px; font-weight: 600; color: var(--text);
  margin-bottom: 12px; letter-spacing: -.2px;
}
.ml-card p { font-size: 14px; color: var(--text2); line-height: 1.75; }
.ml-card-icon-row {
  display: flex; align-items: center; gap: 12px;
  margin-bottom: 24px;
}
.ml-card-icon-box {
  width: 40px; height: 40px; border-radius: 10px;
  background: var(--lime-dim);
  border: 1px solid var(--lime-b);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.ml-card-icon-box svg { width: 20px; height: 20px; }

/* ────────────────── HOW IT WORKS ────────────────── */
.ml-how {
  padding: 100px 7vw;
  background: var(--surface);
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}
.ml-how-title {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-weight: 700; font-style: italic;
  font-size: clamp(32px, 4vw, 52px);
  color: var(--text); margin-bottom: 64px; line-height: 1.1;
  max-width: 400px;
}
.ml-steps {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 0; position: relative;
}
.ml-steps::before {
  content: ''; position: absolute;
  top: 20px; left: 10%; right: 10%; height: 1px;
  background: var(--border);
}
.ml-step { padding-right: 48px; }
.ml-step-num {
  font-family: 'Space Mono', monospace;
  font-size: 11px; color: var(--lime); letter-spacing: 1px;
  margin-bottom: 32px; display: flex; align-items: center; gap: 12px;
}
.ml-step-num::after {
  content: ''; flex: 1; height: 1px;
  background: var(--lime); max-width: 40px; opacity: .3;
}
.ml-step h4 {
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 17px; font-weight: 600; color: var(--text);
  margin-bottom: 10px; letter-spacing: -.2px;
}
.ml-step p { font-size: 14px; color: var(--text2); line-height: 1.75; }

/* ────────────────── BOTTOM CTA ────────────────── */
.ml-cta-section {
  padding: 120px 7vw 100px;
  background: var(--bg);
  display: flex; flex-direction: column;
  align-items: center; text-align: center;
  border-top: 1px solid var(--border);
}
.ml-cta-section h2 {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-weight: 700; font-style: italic;
  font-size: clamp(42px, 6vw, 80px);
  color: var(--text); line-height: 1.02; margin-bottom: 16px;
  max-width: 800px;
}
.ml-cta-section p { color: var(--text2); font-size: 15px; margin-bottom: 40px; font-weight: 300; }

/* ────────────────── FOOTER ────────────────── */
footer.ml-footer, .ml-footer {
  padding: 32px 7vw;
  border-top: 1px solid var(--border);
  display: flex !important; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 12px;
  background: var(--surface);
  width: 100%;
}
.ml-footer-left { font-size: 13px; color: var(--text3); }
.ml-footer-right { font-size: 12px; color: var(--text3); font-style: italic; }
.ml-footer a { color: var(--text2) !important; text-decoration: none !important; }
.ml-footer a:hover { color: var(--text) !important; text-decoration: none !important; }
.ml-nav-links a { text-decoration: none !important; }
.see-how-btn { text-decoration: none !important; }

/* ────────────────── SCANNER ────────────────── */
.ml-scanner-page [data-testid="stMainBlockContainer"],
.ml-scanner-page .block-container {
  padding: 0 max(24px, 5vw) 60px !important;
  max-width: 100% !important;
}
.ml-scanner-header { margin: 8px 0 36px; }
.ml-scanner-title {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-style: italic; font-weight: 700;
  font-size: clamp(32px, 3.5vw, 52px); color: var(--text);
  margin-bottom: 8px; line-height: 1.05;
}
.ml-scanner-desc { font-size: 14px; color: var(--text2); font-weight: 300; line-height: 1.7; }
.ml-label {
  font-family: 'Space Mono', monospace;
  font-size: 10px; color: var(--text3); letter-spacing: 1.5px;
  text-transform: uppercase; margin-bottom: 10px;
}
/* Override Streamlit blue info/success/warning in scanner */
[data-testid="stNotification"],
[data-testid="stAlertContainer"],
div[data-testid="stInfo"],
div.stInfo, div[class*="stAlert"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
}
div[data-testid="stInfo"] svg,
div[class*="stAlert"] svg { color: var(--lime) !important; stroke: var(--lime) !important; }

/* ────────────────── STREAMLIT WIDGETS ────────────────── */
/* Primary buttons — layout on the button element only */
.stButton > button,
[data-testid="stBaseButton-primary"] {
  background: var(--lime) !important;
  color: #07080D !important;
  border: none !important; border-radius: 10px !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 600 !important; font-size: 14px !important;
  letter-spacing: -.1px !important;
  transition: opacity .18s, transform .18s !important;
  padding: 0 24px !important; height: 46px !important;
  white-space: nowrap !important;
}
/* Force dark text on every child node — color only, no layout */
.stButton > button *,
[data-testid="stBaseButton-primary"] * {
  color: #07080D !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 14px !important; font-weight: 600 !important;
}
.stButton > button:hover,
[data-testid="stBaseButton-primary"]:hover {
  opacity: .88 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 24px rgba(200,255,87,0.22) !important;
}
/* Secondary / ghost buttons */
.stButton > button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
  background: transparent !important;
  color: var(--text2) !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important; height: 46px !important;
  white-space: nowrap !important;
}
.stButton > button[kind="secondary"] *,
[data-testid="stBaseButton-secondary"] * { color: var(--text2) !important; }
.stButton > button[kind="secondary"]:hover,
[data-testid="stBaseButton-secondary"]:hover {
  border-color: rgba(255,255,255,0.2) !important;
  color: var(--text) !important;
  background: rgba(255,255,255,0.03) !important;
  transform: none !important; box-shadow: none !important;
}
/* Back button */
.back-btn button {
  background: transparent !important;
  color: var(--text2) !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important; white-space: nowrap !important;
  font-size: 13px !important; font-weight: 500 !important;
  padding: 0 16px !important; height: 36px !important; border-radius: 8px !important;
  transform: none !important;
}
.back-btn button * { color: var(--text2) !important; }
.back-btn button:hover { color: var(--text) !important; border-color: rgba(255,255,255,0.2) !important; background: rgba(255,255,255,0.04) !important; }
.back-btn button:hover * { color: var(--text) !important; }

[data-testid="stFileUploaderDropzone"] {
  border: 1px dashed rgba(255,255,255,0.12) !important;
  border-radius: 12px !important;
  background: var(--surface) !important;
  transition: border-color .2s, background .2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--lime-b) !important;
  background: var(--lime-dim) !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0; background: transparent;
  border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 500 !important; font-size: 13px !important;
  color: var(--text2) !important;
  background: transparent; border: none; padding: 10px 20px;
  letter-spacing: .1px;
}
.stTabs [aria-selected="true"] {
  color: var(--text) !important;
  border-bottom: 2px solid var(--lime) !important;
}

[data-testid="stMetric"] {
  background: var(--surface) !important;
  border-radius: 12px; padding: 20px 22px;
  border: 1px solid var(--border) !important;
}
[data-testid="stMetric"] label {
  color: var(--text2) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 10px !important; letter-spacing: 1px;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 700 !important; font-size: 22px !important;
}

.stTextArea textarea, [data-testid="stChatInput"] textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important; color: var(--text) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stTextArea textarea:focus, [data-testid="stChatInput"] textarea:focus {
  border-color: var(--lime-b) !important;
  box-shadow: 0 0 0 2px rgba(200,255,87,0.1) !important;
}

.stRadio label, .stRadio p {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  color: var(--text2) !important;
}
div[data-baseweb="radio"] > label > div:first-child > div {
  border-color: var(--lime) !important;
}
div[data-baseweb="radio"] > label {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 10px 14px !important; margin-bottom: 6px !important;
  transition: border-color .2s, background .2s;
}
div[data-baseweb="radio"] > label:hover {
  border-color: var(--lime-b) !important;
  background: var(--lime-dim) !important;
}

.stAlert, .element-container .stSuccess,
.element-container .stInfo,
.element-container .stWarning,
.element-container .stError {
  border-radius: 12px !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stSpinner p {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
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
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}
div[data-testid="stDivider"] { border-color: rgba(255,255,255,0.07) !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(200,255,87,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(200,255,87,0.4); }

/* ── Dark theme: patch all remaining Streamlit light-bg leaks ── */

/* Root containers */
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.stApp, .main {
  background-color: #07080D !important;
}

/* Block container — Streamlit adds its own padding/bg */
[data-testid="stMainBlockContainer"],
[data-testid="block-container"],
.block-container {
  background: transparent !important;
  padding: 0 !important;
  max-width: 100% !important;
}

/* Every st.write / st.markdown paragraph — exclude button internals */
[data-testid="stMarkdownContainer"] p:not(.stButton p),
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] code,
.stMarkdown p, .stMarkdown li {
  color: #F0F4FF !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}
/* Re-assert dark text inside primary buttons after any p override */
.stButton > button p,
[data-testid="stBaseButton-primary"] p { color: #07080D !important; }

/* st.write plain text */
[data-testid="stText"] {
  color: #F0F4FF !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Widget labels */
label, .stSelectbox label, .stRadio label,
[data-testid="stWidgetLabel"] p,
[data-baseweb="label"] {
  color: #8892A4 !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* st.info / st.success / st.warning / st.error */
[data-testid="stNotification"],
[data-testid="stAlert"] {
  background: #0E0F16 !important;
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  color: #F0F4FF !important;
}
[data-testid="stAlert"] p { color: #F0F4FF !important; }

/* st.caption */
[data-testid="stCaptionContainer"] p,
.stCaption, .stCaption p {
  color: #4B5A6E !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Text area */
.stTextArea label { color: #8892A4 !important; }
.stTextArea textarea {
  background: #0E0F16 !important;
  color: #F0F4FF !important;
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
  color: #F0F4FF !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-panel"] {
  background: transparent !important;
  padding-top: 20px;
}

/* Chat input area */
[data-testid="stChatInputContainer"] {
  background: #07080D !important;
  border-top: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stChatInput"] {
  background: #0E0F16 !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 14px !important;
}

/* Chat message avatars and text */
[data-testid="stChatMessage"] p { color: #F0F4FF !important; }

/* Metric delta */
[data-testid="stMetricDelta"] svg { stroke: var(--lime) !important; }

/* st.divider */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* Spinner text */
[data-testid="stSpinner"] p { color: #8892A4 !important; }

/* Selectbox / dropdown */
[data-baseweb="select"] > div {
  background: #0E0F16 !important;
  border-color: rgba(255,255,255,0.08) !important;
  color: #F0F4FF !important;
}
[data-baseweb="popover"] ul {
  background: #0E0F16 !important;
  color: #F0F4FF !important;
}

/* Code blocks */
code, pre {
  background: var(--surface) !important;
  color: var(--lime) !important;
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
    <div class="ml-logo-mark">
      <svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
        <rect x="7.5" y="1.5" width="5" height="17" rx="2.5" fill="#07080D"/>
        <rect x="1.5" y="7.5" width="17" height="5" rx="2.5" fill="#07080D"/>
      </svg>
    </div>
    <div class="ml-logo-wordmark">
      <span class="ml-logo-med">Med</span><span class="ml-logo-label">Label</span>
    </div>
  </a>
  <div class="ml-nav-links">
    <a href="#" data-scrollto="ml-features">Features</a>
    <a href="#" data-scrollto="ml-how">How it works</a>
    <a href="#" data-scrollto="ml-safety">Get Started</a>
  </div>
</nav>""", unsafe_allow_html=True)

    components.html("""<script>
(function(){
  function wire(){
    var doc=parent.document;
    doc.querySelectorAll('a[data-scrollto]').forEach(function(a){
      if(a._wired) return;
      a._wired=true;
      a.addEventListener('click',function(e){
        e.preventDefault();
        var id=a.getAttribute('data-scrollto');
        var el=doc.getElementById(id);
        if(el) el.scrollIntoView({behavior:'smooth',block:'start'});
      });
    });
  }
  setTimeout(wire,600);
  setTimeout(wire,1500);
})();
</script>""", height=0)

    st.markdown("""
<div class="ml-hero-wrap">
  <div class="ml-hero-noise"></div>
  <div class="ml-hero-glow"></div>
  <div class="ml-hero-inner">
    <div class="ml-eyebrow">
      <span class="ml-eyebrow-line"></span>
      FDA-grounded &nbsp;·&nbsp; free &nbsp;·&nbsp; private
    </div>
    <div class="ml-hero">
      <h1>Read any medicine label<br><em>in plain English</em></h1>
      <p class="ml-hero-sub">
        Point your phone at any pill bottle, box, or blister pack. MedLabel reads it,
        explains it clearly, and answers your questions — grounded in official FDA data.
      </p>
    </div>
  </div>
  <div class="ml-hero-divider"></div>
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
<div class="ml-stats">
  <div class="ml-stat">
    <div class="ml-stat-num">5,976</div>
    <div class="ml-stat-label">FDA label chunks</div>
  </div>
  <div class="ml-stat">
    <div class="ml-stat-num">3-in-1</div>
    <div class="ml-stat-label">OCR + Vision + YOLO</div>
  </div>
  <div class="ml-stat">
    <div class="ml-stat-num">BGE-M3</div>
    <div class="ml-stat-label">Neural semantic search</div>
  </div>
  <div class="ml-stat">
    <div class="ml-stat-num">0 kb</div>
    <div class="ml-stat-label">Data stored about you</div>
  </div>
</div>

<section class="ml-features" id="ml-features">
  <div class="ml-section-tag">Capabilities</div>
  <div class="ml-features-title">Built for trust<br>and clarity</div>
  <div class="ml-cards">
    <div class="ml-card">
      <div class="ml-card-num">01</div>
      <div class="ml-card-icon-row">
        <div class="ml-card-icon-box">
          <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M6 14V8a2 2 0 0 1 2-2h6" stroke="#C8FF57" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M34 14V8a2 2 0 0 0-2-2h-6" stroke="#C8FF57" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M6 26v6a2 2 0 0 0 2 2h6" stroke="#C8FF57" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M34 26v6a2 2 0 0 1-2 2h-6" stroke="#C8FF57" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
            <line x1="6" y1="20" x2="34" y2="20" stroke="#C8FF57" stroke-width="1.8" stroke-linecap="round" stroke-dasharray="3 2"/>
            <circle cx="20" cy="20" r="2.5" fill="#C8FF57"/>
          </svg>
        </div>
      </div>
      <h3>Smart scanning</h3>
      <p>YOLO geometry detection routes each photo to the right pipeline — OCR for flat labels, Vision for curved bottles.</p>
    </div>
    <div class="ml-card">
      <div class="ml-card-num">02</div>
      <div class="ml-card-icon-row">
        <div class="ml-card-icon-box">
          <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M8 10a3 3 0 0 1 3-3h18a3 3 0 0 1 3 3v13a3 3 0 0 1-3 3H22l-5 5v-5H11a3 3 0 0 1-3-3V10Z" stroke="#C8FF57" stroke-width="2" stroke-linejoin="round"/>
            <line x1="13" y1="15" x2="27" y2="15" stroke="#C8FF57" stroke-width="1.8" stroke-linecap="round"/>
            <line x1="13" y1="20" x2="23" y2="20" stroke="#C8FF57" stroke-width="1.8" stroke-linecap="round" opacity="0.6"/>
          </svg>
        </div>
      </div>
      <h3>Plain-English answers</h3>
      <p>Medical jargon simplified to a 5th-grade reading level. Ask anything — side effects, dosage, interactions — in natural language.</p>
    </div>
    <div class="ml-card">
      <div class="ml-card-num">03</div>
      <div class="ml-card-icon-row">
        <div class="ml-card-icon-box">
          <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M20 4L8 9v10c0 7.18 5.12 13.9 12 15.5C27.88 32.9 33 26.18 33 19V9L20 4Z" stroke="#C8FF57" stroke-width="2" stroke-linejoin="round"/>
            <path d="M14 20l4 4 8-8" stroke="#C8FF57" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
      </div>
      <h3>Grounded in FDA data</h3>
      <p>Every answer is retrieved from 5,976 chunks of official FDA drug labels — never hallucinated, always cited.</p>
    </div>
  </div>
</section>

<section class="ml-how" id="ml-how">
  <div class="ml-section-tag">Process</div>
  <div class="ml-how-title">Three steps<br>to clarity</div>
  <div class="ml-steps">
    <div class="ml-step">
      <div class="ml-step-num">01</div>
      <h4>Upload a photo</h4>
      <p>Take a picture of any medicine label — pill bottle, box, or blister pack.</p>
    </div>
    <div class="ml-step">
      <div class="ml-step-num">02</div>
      <h4>AI reads the label</h4>
      <p>YOLO detects geometry, then our model extracts the text accurately.</p>
    </div>
    <div class="ml-step">
      <div class="ml-step-num">03</div>
      <h4>Get plain answers</h4>
      <p>Ask questions in natural language. Our model answers using official FDA data.</p>
    </div>
  </div>
</section>""", unsafe_allow_html=True)

    st.markdown("""
<div class="ml-cta-section">
  <h2>Ready to scan<br>your first label?</h2>
  <p>Takes under 10 seconds. No account needed.</p>
</div>""", unsafe_allow_html=True)

    _, cta_col, _ = st.columns([3, 2, 3])
    with cta_col:
        if st.button("Get started — scan a label", key="bottom_cta", type="primary", use_container_width=True):
            st.session_state["show_scanner"] = True
            st.rerun()

    st.markdown("""
<footer class="ml-footer" id="ml-safety">
  <div class="ml-footer-left">&#169; 2026 MedLabel &nbsp;&middot;&nbsp; <a href="#">Privacy</a> &nbsp;&middot;&nbsp; <a href="#">Terms</a></div>
  <div class="ml-footer-right">Informational only &#8212; not a substitute for professional medical advice. Always verify with your pharmacist.</div>
</footer>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SCANNER
# ─────────────────────────────────────────────────────────────────────────────
else:
    xai_status = "Set" if os.getenv("XAI_API_KEY") else "Missing"

    st.markdown("""
<nav class="ml-nav">
  <a class="ml-logo" href="#">
    <div class="ml-logo-mark">
      <svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
        <rect x="7.5" y="1.5" width="5" height="17" rx="2.5" fill="#07080D"/>
        <rect x="1.5" y="7.5" width="17" height="5" rx="2.5" fill="#07080D"/>
      </svg>
    </div>
    <div class="ml-logo-wordmark">
      <span class="ml-logo-med">Med</span><span class="ml-logo-label">Label</span>
    </div>
  </a>
  <div class="ml-nav-links">
    <span style="font-family:'Space Mono',monospace;font-size:10px;letter-spacing:1px;color:var(--text3);text-transform:uppercase;">
      AI can make mistakes &#8212; verify with your pharmacist
    </span>
  </div>
</nav>""", unsafe_allow_html=True)

    st.markdown("""
<style>
[data-testid="stMainBlockContainer"], .block-container {
  padding: 0 max(24px, 5vw) 60px !important;
  max-width: 100% !important;
}
</style>""", unsafe_allow_html=True)

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
                "cylindrical": "\U0001fad9  Bottle / curved label",
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

                    spinner_label = {
                        "cylindrical": "Reading bottle label…",
                        "flat":        "Reading flat label…",
                        "auto":        "Analysing label…",
                    }.get(packaging_type, "Analysing label…")
                    with st.spinner(spinner_label):
                        from vision.ocr import run_ocr
                        result = run_ocr(tmp_path, packaging_type=packaging_type)

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
                    st.error(f"A required library is missing: `{exc}` — check your installation.")
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
                    if xai_status == "Missing":
                        st.warning(
                            "xAI API key not set — add `XAI_API_KEY` to your `.env` file."
                        )
                    else:
                        if st.button("Generate Plain-English Summary", type="primary"):
                            with st.spinner("Simplifying label…"):
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

            # Render all messages from history first, so input always sits below.
            for msg in st.session_state.get("chat_history", []):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg["role"] == "assistant":
                        if msg.get("route"):
                            st.caption(f"Route: `{msg['route']}`")
                        if msg.get("results"):
                            with st.expander(f"{len(msg['results'])} FDA label sources"):
                                for i, r in enumerate(msg["results"], 1):
                                    meta = r.get("metadata", {})
                                    st.markdown(
                                        f"**{i}. {meta.get('drug_name','?')}** — "
                                        f"*{meta.get('section_type','?')}* "
                                        f"(score {r.get('final_score',0):.2f})"
                                    )
                                    st.write(r["text"][:500] + "…")
                                    st.divider()

            # If there's a pending question (user just submitted), compute the answer
            # now — the user message is already visible above from the history render.
            if st.session_state.get("_chat_pending"):
                pending_q = st.session_state.pop("_chat_pending")
                with st.spinner("Thinking…"):
                    try:
                        ans = answer_question(pending_q, scanned_drug)
                    except Exception as exc:
                        ans = {"kind": "error",
                               "summary": f"Something went wrong: {exc}",
                               "results": []}
                st.session_state["chat_history"].append({
                    "role":    "assistant",
                    "content": ans["summary"],
                    "route":   ans["kind"],
                    "results": ans.get("results", []),
                })
                st.rerun()

            # Input renders after history → always appears below the last answer.
            # On submit: append user message + set pending flag, then rerun immediately
            # so the query is visible before the spinner starts.
            user_input = st.chat_input("e.g. What are the side effects? Can I take this with Advil?")
            if user_input:
                st.session_state.setdefault("chat_history", []).append(
                    {"role": "user", "content": user_input}
                )
                st.session_state["_chat_pending"] = user_input
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
