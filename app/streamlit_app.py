import streamlit as st
import PIL.Image
import os
from dotenv import load_dotenv

# Load stubs (Relative imports based on repo structure)
# from src.vision.detector import YOLORouter
# from src.vision.ocr import HybridOCR
# from src.agent.brain import MedAgent

load_dotenv()

st.set_page_config(page_title="MedLabel", layout="wide")

st.title("💊 MedLabel: Intelligent Medicine Scanner")
st.markdown("🔴 *AI can make mistakes. Always verify with the physical label or your pharmacist.*")

with st.sidebar:
    st.header("Settings")
    api_key_status = "✅ Set" if os.getenv("GEMINI_API_KEY") else "❌ Missing"
    st.write(f"Gemini API Key: {api_key_status}")
    
    st.divider()
    st.write("Current Package Class: **Detecting...**")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Scan Medicine")
    uploaded_file = st.file_uploader("Upload a photo of your medicine", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        img = PIL.Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Label"):
            with st.spinner("Processing through Hybrid Pipeline..."):
                # Week 6 Task: Integrate the full pipeline
                st.success("Analysis Complete!")
                st.info("Path used: Path A (Flat OCR)")

with col2:
    tab1, tab2, tab3 = st.tabs(["📋 Extracted Info", "📝 Simplified Info", "💬 Chatbot"])
    
    with tab1:
        st.subheader("Structured Label Data")
        st.json({
            "drug_name": "Example Medicine",
            "active_ingredient": "Acetaminophen",
            "dosage": "500 mg",
            "warnings": ["Do not exceed 6 caplets in 24 hours"]
        })
        
    with tab2:
        st.subheader("Plain English Summary")
        st.info("Take 1 pill every 6 hours as needed for pain. Do not take more than 6 pills in one day.")
        
    with tab3:
        st.subheader("Ask a Question")
        user_input = st.chat_input("Can I take this with Advil?")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                st.write("Thinking...")
                # Week 6 Task: Connect to LangChain Agent
                st.write(f"I'm checking the FDA label for information about: {user_input}")
