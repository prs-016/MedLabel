import sys
import tempfile
from pathlib import Path

import streamlit as st
import PIL.Image
import os
from dotenv import load_dotenv

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

load_dotenv()

st.set_page_config(page_title="MedLabel", layout="wide")

st.title("MedLabel: Intelligent Medicine Scanner")
st.markdown("*AI can make mistakes. Always verify with the physical label or your pharmacist.*")

with st.sidebar:
    st.header("Settings")
    api_key_status = "Set" if os.getenv("GEMINI_API_KEY") else "Missing"
    st.write(f"Gemini API Key: {api_key_status}")
    st.divider()
    st.write("Current Package Class: **Detecting...**")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Scan Medicine")
    uploaded_file = st.file_uploader(
        "Upload a photo of your medicine label",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file:
        img = PIL.Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Label"):
            with st.spinner("Running PaddleOCR on flat label..."):
                try:
                    from vision.ocr import run_path_a

                    # Save upload to a temp file so cv2 can read it
                    suffix = Path(uploaded_file.name).suffix or ".png"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    result = run_path_a(tmp_path)
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
        user_input = st.chat_input("Can I take this with Advil?")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                st.write("Thinking...")
                st.write(f"I'm checking the FDA label for information about: {user_input}")
