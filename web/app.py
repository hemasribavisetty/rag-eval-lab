import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_FILE = Path("experiments/results.jsonl")

st.set_page_config(page_title="RAG Eval Lab Dashboard", layout="wide")

st.title("🔍 RAG Optimization & Evaluation Platform")
st.markdown(
    """
This system benchmarks Retrieval-Augmented Generation (RAG) pipelines by:

- Comparing chunking strategies
- Evaluating Precision@K and MRR
- Measuring latency tradeoffs
- Logging experiment runs
- Visualizing quality vs efficiency

Built as a Master's AI Systems Project.
"""
)

st.divider()

if not RESULTS_FILE.exists():
    st.error("No results found. Run: python app/experiments/run_experiment.py")
    st.stop()

rows = []
with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

df = pd.DataFrame(rows)

for col in ["Precision@1", "Precision@3", "Precision@5", "MRR", "ingest_s", "index_s", "eval_s"]:
    if col not in df.columns:
        df[col] = None

df["total_s"] = df["ingest_s"] + df["index_s"] + df["eval_s"]

best_run = df.sort_values(["Precision@1", "MRR"], ascending=False).head(1)

st.header("📊 Benchmark Summary")

c1, c2, c3 = st.columns(3)
c1.metric("Best Precision@1", f"{best_run['Precision@1'].values[0]:.3f}")
c2.metric("Best MRR", f"{best_run['MRR'].values[0]:.3f}")
c3.metric("Total Runs", len(df))

st.write("**Best run configuration:**")
st.dataframe(best_run, width="stretch")

st.divider()

st.header("⚙️ Configuration Comparison")

chunk_sizes = sorted(df["chunk_size"].dropna().unique().tolist())
overlaps = sorted(df["overlap"].dropna().unique().tolist())

c1, c2 = st.columns(2)
with c1:
    selected_cs = st.multiselect("Chunk Size", chunk_sizes, default=chunk_sizes)
with c2:
    selected_ov = st.multiselect("Overlap", overlaps, default=overlaps)

fdf = df[(df["chunk_size"].isin(selected_cs)) & (df["overlap"].isin(selected_ov))]

st.dataframe(
    fdf.sort_values(["Precision@1", "MRR"], ascending=False),
    width="stretch"
)

st.download_button(
    label="Download Results as CSV",
    data=fdf.to_csv(index=False),
    file_name="rag_experiment_results.csv",
    mime="text/csv",
)

st.divider()

st.header("📈 Quality vs Latency")

left, right = st.columns(2)

with left:
    st.write("**Precision@1 vs Total Time**")
    st.scatter_chart(fdf, x="total_s", y="Precision@1")

with right:
    st.write("**MRR vs Total Time**")
    st.scatter_chart(fdf, x="total_s", y="MRR")

st.divider()

best_p1 = fdf.sort_values("Precision@1", ascending=False).head(1)
best_mrr = fdf.sort_values("MRR", ascending=False).head(1)

st.header("🧠 Experiment Insights")
st.write(
    f"""
- Best Precision@1: **{best_p1['Precision@1'].values[0]:.3f}** at chunk_size={best_p1['chunk_size'].values[0]}, overlap={best_p1['overlap'].values[0]}
- Best MRR: **{best_mrr['MRR'].values[0]:.3f}** at chunk_size={best_mrr['chunk_size'].values[0]}, overlap={best_mrr['overlap'].values[0]}
- Overlap appears to improve retrieval quality by preserving context across chunk boundaries.
- This dashboard helps compare retrieval quality against indexing and evaluation time.
"""
)