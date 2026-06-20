import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.query import query_rag_pipeline

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="📄",
    layout="centered"
)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("📄 Document Q&A Bot")
st.markdown("Ask questions about your documents. Answers are grounded with citations.")
st.divider()

# ─────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message:
            with st.expander("📚 Sources"):
                for citation in message["citations"]:
                    st.markdown(f"- 📄 {citation}")

# ─────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents..."):

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Get answer from RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents..."):
            try:
                result = query_rag_pipeline(prompt)
                answer = result["answer"]
                citations = result["citations"]

                # Show answer
                st.markdown(answer)

                # Show citations in expander
                with st.expander("📚 Sources"):
                    for citation in citations:
                        st.markdown(f"- 📄 {citation}")

                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.header("📁 Loaded Documents")
    st.markdown("The bot has knowledge from:")
    
    data_path = "./data"
    if os.path.exists(data_path):
        files = os.listdir(data_path)
        for f in files:
            if f.endswith(".pdf"):
                st.markdown(f"- 📕 {f}")
            elif f.endswith(".docx"):
                st.markdown(f"- 📘 {f}")

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Your question is embedded")
    st.markdown("2. Similar chunks are retrieved")
    st.markdown("3. Gemini generates a grounded answer")
    st.markdown("4. Citations are shown from source docs")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()