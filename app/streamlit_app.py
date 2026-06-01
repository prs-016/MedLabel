import sys
import tempfile
from pathlib import Path

import streamlit as st
import PIL.Image
import os
from dotenv import load_dotenv

import re

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

load_dotenv()


def _load_streamlit_secrets() -> None:
    """Map Streamlit Cloud secrets into os.environ for grok_answer / chroma_config."""
    try:
        if hasattr(st, "secrets") and len(st.secrets) > 0:
            for key, value in st.secrets.items():
                if isinstance(value, str):
                    os.environ.setdefault(key, value)
    except Exception:
        pass


_load_streamlit_secrets()

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
_DOSAGE_WORDS = (
    "dose", "dosage", "how much", "how many", "max daily", "maximum daily",
    "per day", "daily limit", "take at once", "every hours",
)


def _clean_scanned_drug(name: str) -> str:
    """OCR drug name may be a brand; normalize to a generic we can query."""
    if not name:
        return ""
    try:
        from vision.brands import extract_product_name_from_ocr_lines

        fixed = extract_product_name_from_ocr_lines(
            [{"text": name, "confidence": 1.0}]
        )
        if fixed:
            name = fixed
    except Exception:
        pass
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


def answer_question(question: str, scanned_drug: str) -> dict:
    """
    Route a user question to the right trained function and return
    {'kind', 'summary', 'results'} where results is a list of ranked chunks.
    """
    from knowledge.query_functions import (
        vector_search, adverse_events, interaction_check,
    )

    drug = _clean_scanned_drug(scanned_drug)
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
        elif any(w in q for w in _DOSAGE_WORDS):
            search_q = (
                f"{question} dosage and administration maximum daily dose "
                "do not exceed"
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


st.set_page_config(page_title="MedLabel", layout="wide")


@st.cache_resource(show_spinner="Loading label database…")
def _init_label_db() -> int:
    from knowledge.bootstrap import ensure_chroma_ready

    return ensure_chroma_ready()


st.title("MedLabel: Intelligent Medicine Scanner")
st.markdown("*AI can make mistakes. Always verify with the physical label or your pharmacist.*")

with st.sidebar:
    st.header("Settings")
    xai_status = "Set" if os.getenv("XAI_API_KEY") else "Missing"
    st.write(f"xAI API Key: **{xai_status}**")
    if xai_status == "Missing":
        st.caption("Add `XAI_API_KEY=...` to your `.env` file for OCR and chat answers.")
    try:
        from knowledge.chroma_config import get_collection_name

        _n = _init_label_db()
        st.caption(f"Label DB: `{get_collection_name()}` — **{_n}** chunks")
    except Exception as exc:
        st.caption(f"Label DB error: {exc}")
    st.divider()
    packaging_type = st.radio(
        "Label type",
        options=["cylindrical", "flat"],
        format_func=lambda x: "Bottle / curved label (xAI Vision)"
        if x == "cylindrical"
        else "Flat label (PaddleOCR)",
        index=0,
        help="Use Bottle for curved medicine bottles like Advil. "
             "Use Flat for boxes, blister packs, or tube labels.",
    )

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Scan Medicine")
    uploaded_file = st.file_uploader(
        "Upload a photo of your medicine label",
        type=["jpg", "jpeg", "png", "heic", "heif"],
    )

    if uploaded_file:
        try:
            from vision.image_io import open_image

            img = open_image(
                data=uploaded_file.getvalue(),
                filename=uploaded_file.name,
            )
            st.image(img, caption="Uploaded Image", use_container_width=True)
        except ImportError as exc:
            st.error(str(exc))
            img = None
        except Exception as exc:
            st.error(f"Could not open image: {exc}")
            img = None

        if img is not None and st.button("Analyze Label"):
            path_label = (
                "xAI Vision on bottle label..."
                if packaging_type == "cylindrical"
                else "PaddleOCR on flat label..."
            )
            with st.spinner(path_label):
                try:
                    from vision.image_io import materialize_upload
                    from vision.ocr import run_ocr

                    tmp_path = materialize_upload(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        for_opencv=(packaging_type == "flat"),
                        for_vision=(packaging_type == "cylindrical"),
                    )

                    result = run_ocr(tmp_path, packaging_type=packaging_type)
                    os.unlink(tmp_path)

                    if "reupload_required" in result.hallucination_flags:
                        conf_pct = int(result.confidence * 100)
                        st.warning(
                            f"Image quality too low to read the label reliably "
                            f"(confidence: {conf_pct}%, threshold: 75%). "
                            "Please retake the photo with better lighting and a "
                            "flat, unobstructed view of the label, then re-upload."
                        )
                    else:
                        st.session_state["ocr_result"] = result
                        st.success(
                            f"Analysis complete (confidence: {int(result.confidence * 100)}%)"
                        )

                except ImportError as exc:
                    st.error(
                        f"PaddleOCR is not installed: {exc}\n\n"
                        "Run: `pip install paddleocr` and "
                        "`pip install -r requirements-paddle.txt`"
                    )
                except Exception as exc:
                    st.error(f"OCR failed: {exc}")

with col2:
    tab1, tab2, tab3 = st.tabs(["Extracted Info", "Simplified Info", "Chatbot"])

    with tab1:
        st.subheader("Structured Label Data")
        result = st.session_state.get("ocr_result")
        if result:
            st.metric("Drug Name", result.drug_name or "—")
            st.metric("Dosage", result.dosage or "—")
            st.divider()
            st.caption("All text detected on label:")
            st.text_area("Raw label text", result.raw_text, height=300, disabled=True)
            st.caption(f"Path used: {result.path_used} | Confidence: {result.confidence:.2%}")
        else:
            st.info("Upload and analyze a label to see results here.")

    with tab2:
        st.subheader("Plain English Summary")
        result = st.session_state.get("ocr_result")
        if result and result.drug_name:
            st.info(
                f"**{result.drug_name}**"
                + (f" — {result.dosage}" if result.dosage else "")
                + "\n\nSee the 'Extracted Info' tab for full label text."
            )
        else:
            st.info("Analyze a label to see a plain-English summary.")

    with tab3:
        st.subheader("Ask a Question")
        result = st.session_state.get("ocr_result")
        scanned_drug = (result.drug_name if result else "") or ""

        if scanned_drug:
            st.caption(f"Answering about your scanned drug: **{scanned_drug}**")
        else:
            st.caption("Tip: scan a label first so I can answer about that drug. "
                       "You can still ask general questions.")

        # render prior turns
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("e.g. What are the side effects? / Can I take this with Advil?")
        if user_input:
            st.session_state.setdefault("chat_history", []).append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Searching labels, reranking, and drafting answer..."):
                    try:
                        ans = answer_question(user_input, scanned_drug)
                    except Exception as exc:  # noqa: BLE001
                        ans = {"kind": "error",
                               "summary": f"Something went wrong: {exc}",
                               "results": []}

                st.write(ans["summary"])
                st.caption(f"Routed to: `{ans['kind']}`")

                if ans["results"]:
                    with st.expander(f"Top {len(ans['results'])} reranked sources"):
                        for i, r in enumerate(ans["results"], 1):
                            meta = r.get("metadata", {})
                            st.markdown(
                                f"**{i}. {meta.get('drug_name', '?')}** — "
                                f"*{meta.get('section_type', '?')}*  "
                                f"(score {r.get('final_score', 0):.2f})"
                            )
                            st.write(r["text"][:500] + "…")
                            st.divider()

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": ans["summary"]}
            )
