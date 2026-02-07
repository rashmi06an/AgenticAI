import streamlit as st
import pandas as pd
import os
from rag_web_agent.evaluate import ExperimentRunner
from rag_web_agent.config import EXPERIMENTS_FILE, FINDINGS_FILE, DEFAULT_CONFIG

st.set_page_config(page_title="RAG + Web Search Agent", layout="wide")

st.title("🔍 Hybrid RAG + Web Search Agent")
st.markdown("""
This agent prefers **Web Search** via Serper API and falls back to **Document-based RAG** 
(FAISS + LangChain) if web results are insufficient.
""")

# Sidebar for Hyperparameters
st.sidebar.header("⚙️ Hyperparameter Tuning")
chunk_size = st.sidebar.slider("Chunk Size", 200, 2000, DEFAULT_CONFIG["chunk_size"])
chunk_overlap_pct = st.sidebar.slider("Chunk Overlap (%)", 0, 10, 10)
chunk_overlap = int(chunk_size * (chunk_overlap_pct / 100))

k = st.sidebar.number_input("Top-k Retrieval", 1, 10, DEFAULT_CONFIG["k"])
temp = st.sidebar.slider("Temperature", 0.0, 1.0, DEFAULT_CONFIG["temperature"])
chain_type = st.sidebar.selectbox("RAG Chain Type", ["stuff", "map_reduce", "refine", "map_rerank"])

config = {
    "chunk_size": chunk_size,
    "chunk_overlap": chunk_overlap,
    "k": k,
    "temperature": temp,
    "chain_type": chain_type
}

# Main Query Section
query = st.text_input("💬 Enter your question:", placeholder="e.g., What is the latest in AI?")

if st.button("Run Agent"):
    if query:
        with st.spinner("Agent is thinking..."):
            runner = ExperimentRunner()
            result = runner.run_query(query, config)
            
            # Display Results
            st.subheader("Results")
            
            # Mode Badge
            mode_color = "blue" if result["mode_used"] == "web" else "green"
            st.markdown(f"**Source Mode:** :{mode_color}[{result['mode_used'].upper()}]")
            
            st.write(result["answer"])
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence", f"{result['self_confidence']:.2f}")
            col2.metric("Answer Length", f"{result['answer_len']} chars")
            col3.metric("Time Taken", f"{result['time_taken']}s")
            
            if result["mode_used"] == "docs":
                st.info(f"Retrieved {result['num_sources']} chunks from local documents.")
    else:
        st.warning("Please enter a query.")

st.divider()

# Experimentation Section
st.header("📊 Experimentation & Results")
if st.button("Run Benchmark Experiments"):
    st.info("Running predefined experiments... check terminal for logs.")
    runner = ExperimentRunner()
    queries = ["What is RAG?", "Latest trends in ML 2024", "How to use FAISS?"]
    configs = [
        {"chunk_size": 500, "chunk_overlap": 50, "k": 3, "chain_type": "stuff"},
        {"chunk_size": 1000, "chunk_overlap": 100, "k": 5, "chain_type": "stuff"}
    ]
    runner.run_experiments(queries, configs)
    st.success("Experiments completed!")

# Display Experiments CSV
if os.path.exists(EXPERIMENTS_FILE):
    st.subheader("Experiment History")
    df = pd.read_csv(EXPERIMENTS_FILE)
    st.dataframe(df.tail(10)) # Show last 10
    
    # Download Button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "experiments.csv", "text/csv")
else:
    st.write("No experiments run yet.")

# Display Findings
if os.path.exists(FINDINGS_FILE):
    st.header("📝 Academic Findings")
    with open(FINDINGS_FILE, "r") as f:
        st.markdown(f.read())
