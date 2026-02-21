# 🔍 RAG Optimization & Evaluation Platform

A full-stack AI system for benchmarking Retrieval-Augmented Generation (RAG) pipelines.

## 🚀 Problem

Most RAG systems lack standardized evaluation, reproducibility, and performance benchmarking.

This platform enables:
- Configurable chunking strategies
- Embedding & FAISS vector search
- Retrieval benchmarking (Precision@K, MRR)
- Experiment tracking
- Latency measurement
- Interactive dashboard visualization

---

## 🧠 Architecture

1. Document Ingestion
2. Chunking (configurable size & overlap)
3. Embedding (SentenceTransformers)
4. FAISS Indexing
5. Retrieval
6. Evaluation Harness
7. Experiment Runner
8. Streamlit Dashboard

---

## 📊 Metrics Implemented

- Precision@1
- Precision@3
- Precision@5
- MRR (Mean Reciprocal Rank)
- Ingestion time
- Index build time
- Evaluation time

---

## 🖥️ Dashboard

Run:

```bash
streamlit run web/app.py


How To Run
python app/ingest/ingest.py
python app/embed/build_index.py
python -m app.eval.eval_retrieval
python app/experiments/run_experiment.py