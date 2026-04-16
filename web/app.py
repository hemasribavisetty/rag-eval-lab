import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_FILE = Path("experiments/results.jsonl")

st.set_page_config(page_title="RAG Eval Lab Dashboard", layout="wide")

st.title(" RAG Optimization & Evaluation Platform")
st.markdown(
    """
This system benchmarks Retrieval-Augmented Generation (RAG) pipelines by:

- Comparing chunking strategies
- Evaluating Precision@K and MRR
- Measuring latency tradeoffs
- Logging experiment runs
- Visualizing quality vs efficiency
- Comparing baseline retrieval with cross-encoder reranking

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

expected_cols = [
    "Precision@1", "Precision@3", "Precision@5", "MRR",
    "Precision@1_rerank", "Precision@3_rerank", "Precision@5_rerank", "MRR_rerank",
    "ingest_s", "index_s", "eval_s"
]

for col in expected_cols:
    if col not in df.columns:
        df[col] = None

df["total_s"] = df["ingest_s"] + df["index_s"] + df["eval_s"]

# -----------------------------
# Filters first
# -----------------------------
st.header(" Configuration Comparison")

chunk_sizes = sorted(df["chunk_size"].dropna().unique().tolist())
overlaps = sorted(df["overlap"].dropna().unique().tolist())

c1, c2 = st.columns(2)
with c1:
    selected_cs = st.multiselect("Chunk Size", chunk_sizes, default=chunk_sizes)
with c2:
    selected_ov = st.multiselect("Overlap", overlaps, default=overlaps)

fdf = df[
    (df["chunk_size"].isin(selected_cs)) &
    (df["overlap"].isin(selected_ov))
].copy()

if fdf.empty:
    st.warning("No runs match the selected filters.")
    st.stop()

# -----------------------------
# Benchmark summary
# -----------------------------
best_run = fdf.sort_values(["Precision@1", "MRR"], ascending=False).head(1)

st.header(" Benchmark Summary")

c1, c2, c3 = st.columns(3)
c1.metric("Best Precision@1", f"{best_run['Precision@1'].values[0]:.3f}")
c2.metric("Best MRR", f"{best_run['MRR'].values[0]:.3f}")
c3.metric("Total Runs", len(fdf))

st.write("**Best run configuration:**")
st.dataframe(best_run, width="stretch")

# -----------------------------
# Evaluation metrics panel
# -----------------------------
st.header(" Evaluation Metrics")

best_row = fdf.sort_values(["Precision@1", "MRR"], ascending=False).iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Precision@1", f"{best_row['Precision@1']:.3f}")
m2.metric("Precision@3", f"{best_row['Precision@3']:.3f}")
m3.metric("Precision@5", f"{best_row['Precision@5']:.3f}")
m4.metric("MRR", f"{best_row['MRR']:.3f}")

st.caption("Current evaluation includes retrieval metrics. Answer-level faithfulness and hallucination scoring can be added in the next phase.")

# -----------------------------
# Best configuration recommendation
# -----------------------------
st.header(" Best Configuration Recommendation")

best_cfg = fdf.sort_values(
    ["Precision@1", "MRR", "total_s"],
    ascending=[False, False, True]
).iloc[0]

st.markdown(
    f"""
**Recommended Configuration**

- **Chunk Size:** {int(best_cfg['chunk_size'])}
- **Overlap:** {int(best_cfg['overlap'])}
- **Precision@1:** {best_cfg['Precision@1']:.3f}
- **MRR:** {best_cfg['MRR']:.3f}
- **Total Pipeline Time:** {best_cfg['total_s']:.3f}s

**Why this is best**
- High top-1 retrieval quality
- Strong ranking quality across queries
- Good quality-speed tradeoff for practical deployment
"""
)

if best_cfg["total_s"] < 10:
    st.success("Recommended for lower-latency deployment settings.")
else:
    st.warning("High quality, but total runtime may need optimization for strict latency targets.")

st.divider()

# -----------------------------
# Reranking impact
# -----------------------------
st.header(" Reranking Impact")

if "Precision@1_rerank" in fdf.columns and fdf["Precision@1_rerank"].notna().any():
    best_rerank = fdf.sort_values(
        ["Precision@1_rerank", "MRR_rerank"],
        ascending=False
    ).iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    base_p1 = best_rerank["Precision@1"]
    rerank_p1 = best_rerank["Precision@1_rerank"]
    base_mrr = best_rerank["MRR"]
    rerank_mrr = best_rerank["MRR_rerank"]

    c1.metric("Baseline P@1", f"{base_p1:.3f}")
    c2.metric("Reranked P@1", f"{rerank_p1:.3f}", delta=f"{rerank_p1 - base_p1:+.3f}")
    c3.metric("Baseline MRR", f"{base_mrr:.3f}")
    c4.metric("Reranked MRR", f"{rerank_mrr:.3f}", delta=f"{rerank_mrr - base_mrr:+.3f}")

    st.write(
        "Cross-encoder reranking improves top-1 retrieval quality by reordering the initial FAISS candidates using a stronger relevance model."
    )

    st.subheader(" Baseline vs Reranked Comparison")

    compare_cols = [
        "chunk_size",
        "overlap",
        "Precision@1",
        "Precision@1_rerank",
        "MRR",
        "MRR_rerank",
        "total_s",
    ]

    available_cols = [c for c in compare_cols if c in fdf.columns]
    st.dataframe(
        fdf[available_cols].sort_values("Precision@1_rerank", ascending=False),
        width="stretch"
    )

    st.subheader(" Reranker Improvement by Configuration")

    chart_df = fdf.copy()
    chart_df["config"] = chart_df["chunk_size"].astype(str) + "_ov" + chart_df["overlap"].astype(str)

    st.bar_chart(
        chart_df.set_index("config")[["Precision@1", "Precision@1_rerank"]],
        width="stretch"
    )
else:
    st.info("No reranker results found yet. Run experiments with reranking enabled to populate this section.")

st.divider()

# -----------------------------
# Runs table + export
# -----------------------------
st.header(" All Runs")

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

# -----------------------------
# Quality vs latency
# -----------------------------
st.header(" Quality vs Latency")

left, right = st.columns(2)

with left:
    st.write("**Precision@1 vs Total Time**")
    st.scatter_chart(fdf, x="total_s", y="Precision@1")

with right:
    st.write("**MRR vs Total Time**")
    st.scatter_chart(fdf, x="total_s", y="MRR")

st.divider()

# -----------------------------
# Insights
# -----------------------------
best_p1 = fdf.sort_values("Precision@1", ascending=False).head(1)
best_mrr = fdf.sort_values("MRR", ascending=False).head(1)

st.header(" Experiment Insights")
st.write(
    f"""
- Best Precision@1: **{best_p1['Precision@1'].values[0]:.3f}** at chunk_size={best_p1['chunk_size'].values[0]}, overlap={best_p1['overlap'].values[0]}
- Best MRR: **{best_mrr['MRR'].values[0]:.3f}** at chunk_size={best_mrr['chunk_size'].values[0]}, overlap={best_mrr['overlap'].values[0]}
- Overlap appears to improve retrieval quality by preserving context across chunk boundaries.
- Cross-encoder reranking can substantially improve top-1 ranking quality over the FAISS baseline.
- This dashboard helps compare retrieval quality against indexing and evaluation time.
"""
)