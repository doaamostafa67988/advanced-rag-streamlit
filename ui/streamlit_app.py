import sys
from pathlib import Path

import streamlit as st

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.generation.chain import FinConRAGPipeline
from src.ingestion.bootstrap import build_index, ensure_index, reset_indexes
from src.ingestion.indexer import Indexer
from src.retrieval.hybrid_search import HybridRetriever

st.set_page_config(
    page_title="FinCon & Regulatory RAG Assistant",
    page_icon="💼",
    layout="wide",
)

SAMPLE_QUERIES = [
    "What is the mandatory timeline for Form 8-K disclosure under SEC Item 1.05 for material cybersecurity incidents?",
    "What are the differences between Scope 1, Scope 2, and Scope 3 emissions under IFRS S2 in 2026?",
    "What are the eligibility criteria and turnover thresholds for EU CSRD Wave 2 compliance in 2026?",
    "What is the Basel III Endgame capital output floor percentage for risk-weighted assets?",
    "What confidence level is required for Expected Shortfall under the FRTB market risk framework?",
]

if not settings.GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Add it in the app's Secrets settings (or .streamlit/secrets.toml).")
    st.stop()


@st.cache_resource(show_spinner="Loading models and index (first start can take a few minutes)...")
def get_resources():
    """Load models once per server process and build the index if it is missing."""
    indexer = Indexer()  # single instance: embedding model is loaded once
    built = ensure_index(indexer)
    retriever = HybridRetriever(indexer=indexer)
    pipeline = FinConRAGPipeline(retriever=retriever)
    return indexer, pipeline, built


indexer, pipeline, built_now = get_resources()

st.title("💼 FinCon & Financial Regulatory Intelligence Assistant")
st.markdown(
    "**Production Hybrid RAG** • Hierarchical Parent-Child Indexing • "
    "Cross-Encoder Reranker • Citation Enforcement"
)
if built_now:
    st.toast("Index built from the bootstrap corpus.", icon="✅")

# Sidebar: controls
st.sidebar.header("⚙️ Configuration")
top_k = st.sidebar.slider("Top-K Rerank Results", min_value=1, max_value=8, value=3)
st.sidebar.caption(
    f"Indexed child chunks: {indexer.collection.count() if indexer.collection else 0}"
)

with st.sidebar.expander("Rebuild index"):
    n_samples = st.number_input(
        "Dataset rows to index", min_value=0, max_value=5000,
        value=settings.BOOTSTRAP_SAMPLES, step=250,
    )
    if st.button("🔄 Rebuild"):
        with st.spinner("Rebuilding indexes..."):
            try:
                reset_indexes(indexer)
                stats = build_index(indexer, int(n_samples))
                pipeline.refresh_parent_store()
                st.success(f"Done: {stats['children']} child chunks from {stats['total_docs']} documents.")
            except Exception as e:
                st.error(f"Rebuild failed: {e}")

st.sidebar.markdown("---")
selected_sample = st.sidebar.selectbox("Sample queries:", [""] + SAMPLE_QUERIES)

user_query = st.text_input(
    "Enter your Financial Controlling / Accounting Question:",
    value=selected_sample or "",
)

if st.button("🔍 Search & Analyze", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Executing Hybrid Retrieval, Reranking & Synthesis..."):
            result = pipeline.run(query=user_query.strip(), top_k_rerank=top_k)

        st.markdown("### 📝 Financial Controlling Synthesis")
        st.markdown(result["answer"])

        entities = result.get("entities", [])
        if entities:
            badges = "  ".join(f"`🏷️ {e['text']} ({e['label']})`" for e in entities)
            st.markdown(f"**Recognized Financial Entities:** {badges}")

        if result.get("expanded_query") and result["expanded_query"] != user_query:
            st.info(f"🔄 **Query Expansion:** `{result['expanded_query']}`")

        st.markdown("---")
        st.markdown("### 📚 Source Attributions")
        sources = result.get("sources", [])
        cols = st.columns(len(sources) if sources else 1)
        for i, src in enumerate(sources):
            cols[i % len(cols)].info(f"📄 **{src}**")

        with st.expander("🔍 Deep Dive: Retrieved Child Chunks & Reranker Scores"):
            for idx, chunk in enumerate(result["retrieved_child_chunks"], start=1):
                st.markdown(f"#### Rank {idx}: `{chunk['id']}`")
                st.markdown(
                    f"**Rerank Score:** `{chunk.get('rerank_score', 0.0):.4f}` | "
                    f"**RRF Score:** `{chunk.get('rrf_score', 0.0):.4f}`"
                )
                st.text_area(f"Child Chunk #{idx}", chunk["text"], height=100, key=f"child_{idx}")
                st.markdown(f"**Parent ID:** `{chunk.get('parent_id')}`")
                st.markdown("---")

        with st.expander("📖 Deep Dive: Full Parent Context Sent to LLM"):
            st.code(result["context_used"], language="markdown")
