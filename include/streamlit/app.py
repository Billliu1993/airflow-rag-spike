"""
Streamlit RAG Demo App

Features:
- Q&A with hybrid retrieval (BM25 + vector search)
- LLM augmentation for answers
- Support for multiple search modes (vector, BM25, hybrid)
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification
from include.streamlit.rag_utils import hybrid_search, generate_answer  # noqa: E402

# Configuration
RAW_DOCUMENTS_PATH = project_root / "include" / "raw_documents"
RAW_DOCUMENTS_PATH.mkdir(exist_ok=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main():
    st.set_page_config(
        page_title="RAG Demo - Financial Documents",
        page_icon="📚",
        layout="wide"
    )
    
    init_session_state()
    
    # Header
    st.title("📚 Financial RAG Demo")
    st.markdown("Ask questions about your documents using hybrid retrieval (BM25 + Vector Search)")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("📊 Statistics")
        doc_count = len(list(RAW_DOCUMENTS_PATH.glob("*.*")))
        st.metric("Documents in Folder", doc_count)
        
        st.info("💡 Add documents to `include/raw_documents/` and trigger the Airflow DAG to process them.")
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        search_mode = st.selectbox(
            "Search Mode",
            ["hybrid", "vector", "bm25"],
            help="Hybrid combines BM25 and vector search"
        )
        
        top_k = st.slider(
            "Results to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of document chunks to retrieve"
        )
        
        llm_model = st.selectbox(
            "LLM Model",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            index=1,
            help="Model for generating answers"
        )
        
        # Store in session state
        st.session_state.search_mode = search_mode
        st.session_state.top_k = top_k
        st.session_state.llm_model = llm_model
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}** (Score: {source['score']:.4f})")
                        st.markdown(f"*{source['filename']} - Chunk {source['chunk_index']}*")
                        st.text(source['content'][:300] + "...")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                try:
                    # Get search mode and top_k from session state
                    search_mode = st.session_state.get("search_mode", "hybrid")
                    top_k = st.session_state.get("top_k", 5)
                    llm_model = st.session_state.get("llm_model", "gpt-4o-mini")
                    
                    # Perform hybrid search
                    search_results = hybrid_search(
                        query=prompt,
                        mode=search_mode,
                        top_k=top_k
                    )
                    
                    if not search_results:
                        response = "❌ No relevant documents found. Please upload documents first."
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                    else:
                        # Generate answer using LLM
                        with st.spinner("💭 Generating answer..."):
                            answer = generate_answer(
                                query=prompt,
                                search_results=search_results,
                                model=llm_model
                            )
                            
                            st.markdown(answer)
                            
                            # Add to chat history with sources
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": search_results
                            })
                            
                            # Show sources
                            with st.expander("📚 Sources"):
                                for i, source in enumerate(search_results, 1):
                                    st.markdown(f"**Source {i}** (Score: {source['score']:.4f})")
                                    st.markdown(f"*{source['filename']} - Chunk {source['chunk_index']}*")
                                    st.text(source['content'][:300] + "...")
                                    st.divider()
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Footer
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        st.markdown(f"**Documents:** {doc_count}")


if __name__ == "__main__":
    main()
