import json
import argparse
from collections import defaultdict
from app.retrieve.retriever import Retriever

EVAL_FILE = "experiments/eval_questions.jsonl"

def load_eval_questions(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def precision_at_k(retrieved_sources, expected_source, k):
    return 1.0 if expected_source in retrieved_sources[:k] else 0.0

def reciprocal_rank(retrieved_sources, expected_source):
    for i, src in enumerate(retrieved_sources, start=1):
        if src == expected_source:
            return 1.0 / i
    return 0.0

def evaluate(run_dir: str, top_k: int = 5, rerank_top_k: int = 0, k_values=(1, 3, 5)):
    index_file = f"{run_dir}/faiss.index"
    meta_file = f"{run_dir}/chunks_meta.json"
    chunk_file = f"{run_dir}/chunks.jsonl"

    retriever = Retriever(index_file=index_file, meta_file=meta_file, chunk_file=chunk_file)
    eval_items = load_eval_questions(EVAL_FILE)

    scores = defaultdict(list)

    for item in eval_items:
        q = item["question"]
        expected = item["expected_source"]

        results = retriever.search(q, top_k=top_k, rerank_top_k=rerank_top_k)
        retrieved_sources = [r["source"] for r in results]

        for k in k_values:
            scores[f"P@{k}"].append(precision_at_k(retrieved_sources, expected, k))
        scores["MRR"].append(reciprocal_rank(retrieved_sources, expected))

    print("\n=== Retrieval Evaluation ===")
    print(f"Run: {run_dir}")
    print(f"Questions: {len(eval_items)}")
    print(f"Rerank top_k: {rerank_top_k}")
    for k in k_values:
        print(f"Precision@{k}: {sum(scores[f'P@{k}'])/len(scores[f'P@{k}']):.3f}")
    print(f"MRR: {sum(scores['MRR'])/len(scores['MRR']):.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="experiments")
    parser.add_argument("--rerank_top_k", type=int, default=0)
    args = parser.parse_args()

    evaluate(args.run_dir, rerank_top_k=args.rerank_top_k)