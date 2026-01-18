"""
Streamlit UI for RAG Q&A System
"""
import sys
import time
sys.path.append('src')

import streamlit as st
from pipeline import RAGPipeline


# Page config
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 5px;
        text-align: center;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .startup-metric {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load RAG pipeline (cached) with timing."""
    timing = {}
    
    start_total = time.time()
    
    # Track individual component loading times
    start = time.time()
    pipeline = RAGPipeline.__new__(RAGPipeline)  # Create instance without __init__
    timing['initialization'] = time.time() - start
    
    # Manual initialization with timing
    start = time.time()
    from document_parser import DocumentParser
    from embeddings import EmbeddingGenerator
    from retriever import VectorStore, BM25Retriever, HybridRetriever, Reranker
    from llm_agent import LLMAgent
    timing['imports'] = time.time() - start
    
    # Document Parser
    start = time.time()
    pipeline.parser = DocumentParser(chunk_size=1024, chunk_overlap=200)
    timing['document_parser'] = time.time() - start
    
    # Embeddings
    start = time.time()
    pipeline.embedder = EmbeddingGenerator()
    timing['embeddings'] = time.time() - start
    
    # Vector Store
    start = time.time()
    if pipeline.embedder.primary_model == 'gemini':
        embedding_dim = 768
    else:
        embedding_dim = pipeline.embedder.local_dimension
    pipeline.vector_store = VectorStore(embedding_dim=embedding_dim)
    timing['vector_store'] = time.time() - start
    
    # BM25
    start = time.time()
    pipeline.bm25_retriever = BM25Retriever()
    timing['bm25_retriever'] = time.time() - start
    
    # Hybrid Retriever
    start = time.time()
    pipeline.hybrid_retriever = HybridRetriever(
        embedder=pipeline.embedder,
        vector_store=pipeline.vector_store,
        bm25_retriever=pipeline.bm25_retriever
    )
    timing['hybrid_retriever'] = time.time() - start
    
    # Reranker
    start = time.time()
    pipeline.reranker = Reranker()
    timing['reranker'] = time.time() - start
    
    # LLM
    start = time.time()
    pipeline.llm = LLMAgent()
    timing['llm_agent'] = time.time() - start
    
    # Set attributes
    pipeline.data_folder = "data"
    pipeline.use_reranking = True
    
    # Ingestion
    start = time.time()
    pipeline.ingest_documents()
    timing['ingestion'] = time.time() - start
    
    timing['total'] = time.time() - start_total
    
    return pipeline, timing


def display_answer(result: dict):
    """Display answer with formatting."""
    # Answer section
    st.markdown("### 💬 Answer")
    
    # Confidence badge
    confidence = result['confidence']
    if confidence == 'high':
        confidence_class = "confidence-high"
        confidence_emoji = "✅"
    elif confidence == 'medium':
        confidence_class = "confidence-medium"
        confidence_emoji = "⚠️"
    else:
        confidence_class = "confidence-low"
        confidence_emoji = "❌"
    
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.markdown(f"""
        <div class="answer-box">
            {result['answer']}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size: 2rem;">{confidence_emoji}</div>
            <div class="{confidence_class}">{confidence.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sources section
    if result['sources']:
        st.markdown(f"### 📚 Sources ({len(result['sources'])} found)")
        
        for i, source in enumerate(result['sources']):
            with st.expander(f"[{source['citation_id']}] {source['doc_name']}", expanded=(i == 0)):
                st.markdown(f"**Excerpt:**")
                st.info(source['excerpt'])
                
                st.markdown("---")
                
                # Show full text (always visible)
                st.markdown(f"**Full Chunk Text:**")
                st.text_area(
                    f"Full text from {source['doc_name']}",
                    value=source['full_text'],
                    height=200,
                    key=f"fulltext_{source['chunk_id']}_{i}_{hash(result['query'])}",
                    disabled=True
                )
    else:
        st.warning("⚠️ No sources found or confidence too low.")
    
    # Metrics section
    st.markdown("### ⏱️ Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Retrieval", f"{result['timing']['retrieval']:.3f}s")
    with col2:
        st.metric("Reranking", f"{result['timing']['reranking']:.3f}s")
    with col3:
        st.metric("LLM", f"{result['timing']['llm']:.3f}s")
    with col4:
        st.metric("Total", f"{result['timing']['total']:.3f}s")


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<p class="main-header">🤖 Document Q&A System</p>', unsafe_allow_html=True)
    st.markdown("Ask questions about your documents. The system will provide answers with sources.")
    
    # Initialize session state
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'query_value' not in st.session_state:
        st.session_state.query_value = ""
    
    # Load pipeline (with timing)
    try:
        with st.spinner("🔧 Initializing RAG system..."):
            pipeline, startup_timing = load_pipeline()
        st.success("✅ System ready! Ask your question below.")
    except Exception as e:
        st.error(f"❌ Error loading system: {e}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Advanced settings
        with st.expander("Advanced Options", expanded=False):
            top_k = st.slider("Results to return", 3, 20, 10, key="top_k_slider")
            retrieval_k = st.slider("Retrieval pool size", 20, 100, 30, key="retrieval_k_slider")
        
        st.markdown("---")
        
        # Startup Metrics
        st.markdown("### ⏱️ Startup Time")
        st.metric("Total Startup Time", f"{startup_timing['total']:.2f}s", 
                 delta=None, delta_color="off")
        
        with st.expander("📊 Breakdown", expanded=False):
            # Sort by time (descending)
            sorted_timing = sorted(
                [(k, v) for k, v in startup_timing.items() if k != 'total'],
                key=lambda x: x[1],
                reverse=True
            )
            
            for component, duration in sorted_timing:
                percentage = (duration / startup_timing['total']) * 100
                st.markdown(f"""
                <div class="startup-metric">
                    <strong>{component.replace('_', ' ').title()}</strong><br>
                    {duration:.3f}s ({percentage:.1f}%)
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System info
        st.markdown("### 📊 System Info")
        st.info("""
        **Pipeline:**
        - Vector Search (Qdrant)
        - BM25 Keyword Search
        - Hybrid Fusion (RRF)
        - Cross-Encoder Reranking
        - LLM Generation (Gemini)
        """)
        
        # Example queries
        st.markdown("### 💡 Example Queries")
        st.caption("Click to use, then press Search")
        
        example_queries = [
            "What is machine learning?",
            "Explain cloud computing services",
            "How do neural networks work?",
            "What is data preprocessing?"
        ]
        
        for idx, example_query in enumerate(example_queries):
            if st.button(example_query, key=f"example_btn_{idx}", use_container_width=True):
                st.session_state.query_value = example_query
                st.rerun()
    
    # Query input with Clear button on the right
    col_input, col_clear = st.columns([0.85, 0.15])
    
    with col_input:
        query = st.text_input(
            "Enter your question:",
            value=st.session_state.query_value,
            placeholder="Type your question here...",
            key="query_input_field",
            label_visibility="visible"
        )
    
    with col_clear:
        st.markdown("<div style='margin-top: 1.85rem;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear", use_container_width=True, key="clear_btn"):
            st.session_state.query_value = ""
            st.session_state.last_result = None
            st.session_state.last_query = ""
            st.rerun()
    
    # Update session state with current query value
    st.session_state.query_value = query
    
    # Search button
    col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True, type="primary", key="search_btn")
    
    # Process query
    if search_button:
        if query and query.strip():
            with st.spinner("🔄 Processing your query..."):
                try:
                    # Execute query
                    result = pipeline.query(
                        query=query,
                        top_k=top_k,
                        retrieval_k=retrieval_k,
                        verbose=False
                    )
                    
                    # Store result and query in session state
                    st.session_state.last_result = result
                    st.session_state.last_query = query
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Please enter a question first!")
    
    # Display results if available
    if st.session_state.last_result is not None:
        st.markdown("---")
        
        # Show which query this answer is for
        st.caption(f"🔍 Answered: {st.session_state.last_query}")
        
        display_answer(st.session_state.last_result)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        Built with Streamlit, Qdrant, and Gemini | RAG System v1.0
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
