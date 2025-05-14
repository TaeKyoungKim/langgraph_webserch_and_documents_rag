import streamlit as st
import os
from typing import List, Dict, Any
from langchain.schema import Document
from improved_web_search import setup_and_run

st.set_page_config(
    page_title="Corrective RAG with Web Search",
    page_icon="🔍",
    layout="wide"
)

# Configure API keys (make sure to set these in your environment or use st.secrets in production)
if "GOOGLE_API_KEY" not in os.environ or os.environ["GOOGLE_API_KEY"] == "YOUR_GOOGLE_API_KEY":
    os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "")

if "TAVILY_API_KEY" not in os.environ or os.environ["TAVILY_API_KEY"] == "YOUR_TAVILY_API_KEY":
    os.environ["TAVILY_API_KEY"] = st.secrets.get("TAVILY_API_KEY", "")

# App title and description
st.title("🔍 Corrective RAG with Web Search")
st.markdown("""
This application uses a corrective RAG (Retrieval-Augmented Generation) system with LangGraph.
When your question can't be answered from the local knowledge base, it automatically performs web search.
""")

# Sidebar for custom URLs
st.sidebar.header("Knowledge Base Settings")
st.sidebar.markdown("You can customize the knowledge base by adding URLs below.")

# Default URLs
default_urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# URL inputs
custom_urls = []
use_default = st.sidebar.checkbox("Use default knowledge base", value=True)

if not use_default:
    st.sidebar.markdown("Enter URLs to use as knowledge base:")
    for i in range(3):  # Allow adding up to 3 custom URLs
        url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
        if url:
            custom_urls.append(url)

# Input for question
user_question = st.text_input("Ask a question:", placeholder="e.g., What is agent memory in LLMs?")

# Display debug info checkbox
show_debug = st.sidebar.checkbox("Show debug information", value=False)

# Create a placeholder for the answer
answer_placeholder = st.empty()

# Create debug info expander (hidden by default)
with st.expander("Debug Information", expanded=show_debug):
    debug_placeholder = st.empty()

# Process the question when submitted
if user_question:
    with st.spinner('Searching and generating answer...'):
        # Determine which URLs to use
        urls_to_use = custom_urls if custom_urls and not use_default else default_urls
        
        # Set up and run the RAG system with modified stdout to capture debug info
        import sys
        import io
        
        # Capture stdout to get debug info
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        try:
            # Run the RAG system
            answer = setup_and_run(user_question, urls_to_use)
            
            # Get debug info
            debug_info = new_stdout.getvalue()
        finally:
            # Restore stdout
            sys.stdout = old_stdout
        
        # Display the answer with markdown formatting
        answer_placeholder.markdown(f"### Answer\n{answer}")
        
        # Display debug info if requested
        if show_debug:
            debug_placeholder.code(debug_info, language="text")

# Footer
st.markdown("---")
st.markdown("Built with LangGraph, LangChain, and Gemini AI") 