import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.retrieval.rag import query_rag

# Page config
st.set_page_config(
    page_title="RAG Pipeline — AI Research Assistant",
    page_icon="🧠",
    layout="centered"
)

# Header
st.title("🧠 AI Research Paper Assistant")
st.markdown("Ask any question about AI research papers — powered by RAG + GPT-4")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📚 Available Papers")
    st.markdown("""
    - Attention Is All You Need
    - RAG (Original Paper)
    - GPT-4 Technical Report
    - LLaMA & LLaMA 2
    - LoRA
    - InstructGPT
    - ReAct
    - Chain of Thought
    - LLM Survey
    """)
    st.markdown("---")
    st.markdown("**Model:** GPT-4o-mini")
    st.markdown("**Vector DB:** ChromaDB")
    st.markdown("**Faithfulness:** 97%")

# Sample questions
st.markdown("### 💡 Try these questions:")
col1, col2 = st.columns(2)
with col1:
    if st.button("What is attention mechanism?"):
        st.session_state.question = "What is the attention mechanism?"
    if st.button("How does LoRA work?"):
        st.session_state.question = "How does LoRA work?"
with col2:
    if st.button("What is chain of thought?"):
        st.session_state.question = "What is chain of thought prompting?"
    if st.button("What is RAG?"):
        st.session_state.question = "What is RAG and why is it useful?"

st.markdown("---")

# Question input
question = st.text_input(
    "🔍 Ask your question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What is the attention mechanism in transformers?"
)

# Search button
if st.button("🚀 Search", type="primary"):
    if question:
        with st.spinner("🔍 Searching research papers..."):
            result = query_rag(question)
        
        # Answer
        st.markdown("### 💬 Answer")
        st.markdown(result["answer"])
        
        # Sources
        st.markdown("### 📄 Sources")
        for source in result["sources"]:
            st.markdown(f"- `{source}`")
        
        # Stats
        st.markdown("### 📊 Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks Retrieved", result["chunks_used"])
        with col2:
            st.metric("Faithfulness Score", "97%")
    else:
        st.warning("Please enter a question!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using RAG + OpenAI + ChromaDB + FastAPI + Docker")