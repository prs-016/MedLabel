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

# Merge Streamlit Cloud secrets into os.environ so all downstream code
# (which reads os.getenv) works identically in local and cloud environments.
for _k, _v in st.secrets.items():
    if isinstance(_v, str):
        os.environ.setdefault(_k, _v)

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
st.set_page_config(
    page_title="MedLabel",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design-system CSS (matches Figma: Inter font, teal #17B8A6, dark #0D2737)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .stApp, .main, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #F5F9FA !important;
}

/* ── Hide Streamlit chrome ── */
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
    background: #ffffff; border-bottom: 1px solid #E5EDF0;
    position: sticky; top: 0; z-index: 1000;
}
.ml-logo {
    display: flex; align-items: center; gap: 10px;
    font-weight: 700; font-size: 17px; color: #0D2737;
    text-decoration: none;
}
.ml-logo-icon {
    width: 32px; height: 32px; background: #17B8A6;
    border-radius: 8px; flex-shrink: 0;
}
.ml-nav-links { display: flex; align-items: center; gap: 28px; }
.ml-nav-links a {
    color: #374151; text-decoration: none;
    font-size: 15px; font-weight: 500; transition: color .15s;
}
.ml-nav-links a:hover { color: #17B8A6; }

/* ── Shared buttons ── */
.ml-btn {
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: 10px; font-weight: 600; font-size: 15px;
    cursor: pointer; text-decoration: none !important;
    transition: all .15s; border: none; padding: 11px 24px;
    font-family: 'Inter', sans-serif;
}
.ml-btn-primary { background: #17B8A6 !important; color: #ffffff !important; }
.ml-btn-primary:hover { background: #0FA898 !important; color: #ffffff !important; }
.ml-btn-outline {
    background: transparent !important; color: #0D2737 !important;
    border: 2px solid #0D2737 !important;
}
.ml-btn-outline:hover { background: #0D2737 !important; color: #ffffff !important; }
.ml-btn-sm { padding: 8px 18px !important; font-size: 14px !important; }

/* ── Hero ── */
.ml-hero {
    text-align: center; padding: 90px 20px 110px;
    background: #F5F9FA;
}
.ml-badge {
    display: inline-flex; align-items: center; gap: 7px;
    color: #17B8A6; font-size: 13px; font-weight: 600;
    letter-spacing: .3px; margin-bottom: 28px;
}
.ml-badge-dot {
    width: 8px; height: 8px; background: #17B8A6;
    border-radius: 50%; display: inline-block;
}
.ml-hero h1 {
    font-size: clamp(36px, 5vw, 60px); font-weight: 800; color: #0D2737;
    line-height: 1.08; max-width: 860px; margin: 0 auto 22px;
    letter-spacing: -.5px;
}
.ml-hero-sub {
    font-size: 18px; color: #6B7C8D; max-width: 520px;
    margin: 0 auto 44px; line-height: 1.65;
}
.ml-hero-btns {
    display: flex; gap: 16px; justify-content: center; flex-wrap: wrap;
}

/* ── Features ── */
.ml-features { padding: 88px 80px; background: #F5F9FA; }
.ml-features-title {
    text-align: center; font-size: clamp(26px, 3vw, 38px);
    font-weight: 800; color: #0D2737; margin-bottom: 52px;
    letter-spacing: -.3px;
}
.ml-cards {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 24px; max-width: 1100px; margin: 0 auto;
}
@media (max-width: 900px) { .ml-cards { grid-template-columns: 1fr; } }
.ml-card {
    background: #ffffff; border-radius: 18px;
    padding: 32px 28px; border: 1px solid #E5EDF0;
    transition: box-shadow .2s;
}
.ml-card:hover { box-shadow: 0 4px 24px rgba(23,184,166,.12); }
.ml-card-icon {
    width: 52px; height: 52px; background: #E8F8F6;
    border-radius: 14px; margin-bottom: 22px;
}
.ml-card h3 {
    font-size: 18px; font-weight: 700; color: #0D2737; margin-bottom: 10px;
}
.ml-card p { font-size: 14px; color: #6B7C8D; line-height: 1.65; }

/* ── Footer ── */
.ml-footer {
    background: #E8F4F2; padding: 36px 80px;
    text-align: center; color: #6B7C8D;
    font-size: 14px; border-top: 1px solid #C8E8E2;
}

/* ── Scanner page ── */
.ml-scanner { max-width: 1280px; margin: 0 auto; padding: 36px 48px; }
.ml-scanner-header { margin-bottom: 32px; }
.ml-scanner-title {
    font-size: 26px; font-weight: 800; color: #0D2737; margin-bottom: 6px;
}
.ml-scanner-desc { font-size: 15px; color: #6B7C8D; }
.ml-label {
    font-size: 11px; font-weight: 700; color: #6B7C8D;
    letter-spacing: .8px; text-transform: uppercase; margin-bottom: 8px;
}

/* ── Streamlit widget overrides ── */
.stButton > button {
    background: #17B8A6 !important; color: #ffffff !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 15px !important; transition: background .15s !important;
}
.stButton > button:hover { background: #0FA898 !important; }
.stButton > button[kind="secondary"] {
    background: transparent !important; color: #0D2737 !important;
    border: 2px solid #E5EDF0 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #17B8A6 !important; color: #17B8A6 !important;
    background: transparent !important;
}
div[data-testid="stBackButton"] button,
.back-btn button {
    background: transparent !important; color: #17B8A6 !important;
    border: none !important; font-size: 14px !important;
    font-weight: 500 !important; padding: 0 !important;
    box-shadow: none !important;
}

[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #17B8A6 !important;
    border-radius: 14px !important; background: #F0FBF9 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: transparent;
    border-bottom: 2px solid #E5EDF0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; color: #6B7C8D !important;
    background: transparent; border: none; padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    color: #17B8A6 !important;
    border-bottom: 2px solid #17B8A6 !important;
}

[data-testid="stMetric"] {
    background: #ffffff; border-radius: 12px;
    padding: 16px 20px; border: 1px solid #E5EDF0;
}

.stTextArea textarea,
[data-testid="stChatInput"] textarea {
    border-radius: 12px !important; border-color: #E5EDF0 !important;
    font-family: 'Inter', sans-serif !important;
}

.stRadio label, .stRadio p {
    font-family: 'Inter', sans-serif !important; color: #374151 !important;
}

div[data-baseweb="radio"] > label > div:first-child > div {
    border-color: #17B8A6 !important;
}

.stAlert, .stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
}

.stSpinner p { font-family: 'Inter', sans-serif !important; color: #6B7C8D !important; }

.stExpander { border: 1px solid #E5EDF0 !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Navigation state  (HTML buttons set ?scan=true; back button uses session_state)
# ─────────────────────────────────────────────────────────────────────────────
if st.query_params.get("scan") == "true":
    st.session_state["show_scanner"] = True
    st.query_params.clear()
    st.rerun()

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
    <a href="#how-it-works">How it works</a>
    <a href="#features">Features</a>
    <a href="#safety">Safety</a>
    <a href="#">Docs</a>
    <a href="#"
       onclick="window.location.search='?scan=true';return false;"
       class="ml-btn ml-btn-primary ml-btn-sm">Scan a label</a>
  </div>
</nav>

<section class="ml-hero" id="how-it-works">
  <div class="ml-badge">
    <span class="ml-badge-dot"></span>Safe
  </div>
  <h1>Understand any medicine label in seconds</h1>
  <p class="ml-hero-sub">
    Point your phone at any pill bottle, box, or blister pack. MedLabel reads it,
    explains it in plain English, and answers your questions — grounded in official FDA data.
  </p>
  <div class="ml-hero-btns">
    <a href="#"
       onclick="window.location.search='?scan=true';return false;"
       class="ml-btn ml-btn-primary">Scan a label</a>
    <a href="#features" class="ml-btn ml-btn-outline">See how it works</a>
  </div>
</section>

<section class="ml-features" id="features">
  <div class="ml-features-title">Built for trust and clarity</div>
  <div class="ml-cards">
    <div class="ml-card">
      <div class="ml-card-icon"></div>
      <h3>Smart scanning</h3>
      <p>YOLO routes each photo to the right reader — OCR for flat labels, Vision AI for curved bottles.</p>
    </div>
    <div class="ml-card">
      <div class="ml-card-icon"></div>
      <h3>Plain-English answers</h3>
      <p>Medical jargon simplified to a 5th-grade reading level, every time.</p>
    </div>
    <div class="ml-card">
      <div class="ml-card-icon"></div>
      <h3>Grounded in FDA data</h3>
      <p>Every answer is cited from official FDA drug labels — never hallucinated.</p>
    </div>
  </div>
</section>

<footer class="ml-footer" id="safety">
  <p>© 2026 MedLabel &nbsp;·&nbsp; An informational tool, not a substitute for professional medical advice.</p>
</footer>
""", unsafe_allow_html=True)

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
    <span style="font-size:13px;color:#6B7C8D;font-family:Inter,sans-serif;">
      AI can make mistakes — always verify with your pharmacist.
    </span>
  </div>
</nav>
""", unsafe_allow_html=True)

    st.markdown('<div class="ml-scanner">', unsafe_allow_html=True)

    # Back button + page title
    _bc, _tc = st.columns([1, 7])
    with _bc:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Home", key="back_home"):
            st.session_state["show_scanner"] = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="ml-scanner-header">
  <div class="ml-scanner-title">Scan your medicine label</div>
  <div class="ml-scanner-desc">Upload a photo — MedLabel reads it, simplifies it, and answers your questions.</div>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="ml-label">Label type</div>', unsafe_allow_html=True)
        packaging_type = st.radio(
            "Label type",
            options=["cylindrical", "flat", "auto"],
            format_func=lambda x: {
                "cylindrical": "🫙  Bottle / curved label  (xAI Vision)",
                "flat":        "📦  Flat label  (PaddleOCR)",
                "auto":        "🤖  Auto-detect  (YOLO)",
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
            img = PIL.Image.open(uploaded_file)
            st.image(img, caption="Uploaded image", use_container_width=True)

            if st.button("Analyze Label", type="primary", use_container_width=True):
                suffix = Path(uploaded_file.name).suffix or ".png"
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
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

    st.markdown('</div>', unsafe_allow_html=True)  # close ml-scanner
