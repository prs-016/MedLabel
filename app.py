import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Import your custom RAG pipeline!
from src.agent.tools import interaction_check

# --- Configuration ---
load_dotenv()
# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# We will use Gemini 3.5 Flash as it is incredibly fast and great for RAG
model = genai.GenerativeModel('gemini-3.5-flash')

# --- Streamlit UI Setup ---
st.set_page_config(page_title="MedLabel AI", page_icon="💊", layout="centered")

st.title("💊 MedLabel Assistant")
st.caption("Ask questions about drug interactions powered by FDA data and a custom Cross-Encoder RAG pipeline.")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chatbot Logic ---
# Wait for the user to type a question and hit enter
if user_query := st.chat_input("e.g., Can I take warfarin safely with amoxicillin?"):
    
    # 1. Display the user's question on the screen
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Save the user's question to history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Process the answer
    with st.chat_message("assistant"):
        # Show a loading spinner while your backend does the heavy lifting
        with st.spinner("Searching FDA labels and reranking interactions..."):
            
            # Step A: Run your custom Two-Stage RAG Pipeline
            best_chunks = interaction_check(user_query)
            
            # Format the chunks into a single readable block of text
            if isinstance(best_chunks, list):
                context_text = "\n\n".join(best_chunks)
            else:
                context_text = best_chunks # Fallback if no chunks were found

        with st.spinner("Synthesizing final answer..."):
            # Step B: The System Prompt for Gemini
            # This forces Gemini to ONLY use your FDA data, preventing AI hallucinations.
            prompt = f"""
            You are an expert medical assistant. You will be provided with a user's question 
            and a set of highly relevant, verified medical text chunks extracted from an FDA database.
            
            Your job is to answer the user's question using ONLY the provided context. 
            Do not use outside knowledge. If the context does not contain the answer, explicitly state: 
            "I do not have enough information in my database to answer this."
            
            USER QUESTION: {user_query}
            
            VERIFIED CONTEXT:
            {context_text}
            """

            # Step C: Send to Gemini and get the response
            response = model.generate_content(prompt)
            final_answer = response.text

        # 3. Display the final answer on the screen
        st.markdown(final_answer)
        
        # Optional: Add an expander to show the user the raw data the AI used
        with st.expander("🔍 View Raw Database Hits (Cross-Encoder Output)"):
            st.write(context_text)

    # Save the assistant's response to history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})