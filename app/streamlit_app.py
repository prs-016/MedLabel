"""
MedLabel demo: upload label photo → OCR → chat with interaction_check agent.

Run from repo root::

    export PYTHONPATH=src
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(ROOT / ".env")

from agent.interaction_check import interaction_check
from agent.tools import adverse_events, interaction_check as tool_interaction_check
from api.xai_client import get_xai_api_key
from pipeline.scan import ocr_result_to_dict, resolve_scanned_drug, scan_image


def _llm_status() -> str:
    if get_xai_api_key():
        return "xAI"
    if __import__("os").getenv("GEMINI_API_KEY"):
        return "Gemini"
    return "missing"


def _is_interaction_question(question: str) -> bool:
    q = question.lower()
    return any(
        w in q for w in ("with", "together", "combine", "interaction", "mix")
    ) or ("take" in q and "with" in q)


def _is_adverse_events_question(question: str) -> bool:
    q = question.lower()
    if _is_interaction_question(question):
        return False
    return any(
        w in q
        for w in (
            "side effect",
            "side effects",
            "adverse",
            "symptom",
            "symptoms",
            "faers",
            "reaction",
            "reactions",
        )
    )


def _answer_question(question: str, scanned_drug: str) -> str:
    """Answer chat using interaction_check / adverse_events tools."""
    if _is_adverse_events_question(question):
        return adverse_events(question, scanned_drug=scanned_drug)

    if _is_interaction_question(question):
        try:
            return tool_interaction_check(question, scanned_drug=scanned_drug)
        except Exception:
            return interaction_check(question, scanned_drug=scanned_drug).format_for_agent()

    return (
        f"Scanned drug: **{scanned_drug}**. Ask about:\n"
        "- Drug interactions: *Can I take this with ibuprofen?*\n"
        "- Side effects: *What are the side effects of this medicine?*"
    )


st.set_page_config(page_title="MedLabel", layout="wide")

st.title("MedLabel: Intelligent Medicine Scanner")
st.caption(
    "Upload a label photo, then ask questions. "
    "Drug interaction answers use DDInter + FDA label data."
)
st.markdown(
    "🔴 *AI can make mistakes. Always verify with the physical label or your pharmacist.*"
)

if "scan_payload" not in st.session_state:
    st.session_state.scan_payload = None
if "scanned_drug" not in st.session_state:
    st.session_state.scanned_drug = ""
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

with st.sidebar:
    st.header("Settings")
    st.write(f"LLM / tools: **{_llm_status()}**")
    packaging = st.radio(
        "Label type",
        options=["flat", "cylindrical"],
        format_func=lambda x: "Flat box / blister (PaddleOCR)" if x == "flat" else "Bottle (xAI Vision)",
        help="Path A for flat packaging, Path B for curved bottles.",
    )
    st.divider()
    if st.session_state.scanned_drug:
        st.success(f"Scanned drug: **{st.session_state.scanned_drug}**")
    else:
        st.info("Upload and analyze a label to enable chat.")

col_upload, col_results = st.columns([1, 1])

with col_upload:
    st.header("Scan medicine")
    uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        st.image(Image.open(uploaded), caption="Uploaded image", use_container_width=True)

        if st.button("Analyze label", type="primary"):
            suffix = Path(uploaded.name or "upload.jpg").suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            with st.spinner("Running OCR…"):
                try:
                    ocr_result = scan_image(tmp_path, packaging)  # type: ignore[arg-type]
                    scanned = resolve_scanned_drug(ocr_result)
                    payload = ocr_result_to_dict(ocr_result, scanned_drug=scanned)
                    st.session_state.scan_payload = payload
                    st.session_state.scanned_drug = scanned
                    st.session_state.chat_messages = []
                except Exception as exc:
                    st.error(f"OCR failed: {exc}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

with col_results:
    tab_info, tab_chat = st.tabs(["Extracted label", "Chat"])

    with tab_info:
        if st.session_state.scan_payload:
            p = st.session_state.scan_payload
            st.subheader("Structured fields")
            st.json(
                {
                    "scanned_drug": p.get("scanned_drug"),
                    "drug_name": p.get("drug_name"),
                    "dosage": p.get("dosage"),
                    "active_ingredients": p.get("active_ingredients"),
                    "path_used": p.get("path_used"),
                    "confidence": p.get("confidence"),
                    "hallucination_flags": p.get("hallucination_flags"),
                }
            )
            if p.get("hallucination_flags"):
                st.warning(f"Quality flags: {', '.join(p['hallucination_flags'])}")
            with st.expander("Raw OCR text"):
                st.text(p.get("raw_text_preview") or "")
        else:
            st.info("Analyze a label to see extracted fields.")

    with tab_chat:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("e.g. Can I take this with ibuprofen?")
        if prompt:
            if not st.session_state.scanned_drug:
                st.warning("Analyze a label first so we know which medicine you scanned.")
            else:
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Checking…"):
                        reply = _answer_question(
                            prompt,
                            st.session_state.scanned_drug,
                        )
                    st.markdown(reply)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": reply}
                )
