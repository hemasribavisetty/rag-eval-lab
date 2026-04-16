from app.retrieve.retriever import Retriever


def generate_answer(query: str, run_dir: str, rerank_top_k: int = 5):
    retriever = Retriever(
        index_file=f"{run_dir}/faiss.index",
        meta_file=f"{run_dir}/chunks_meta.json",
        chunk_file=f"{run_dir}/chunks.jsonl",
    )

    results = retriever.search(query, top_k=3, rerank_top_k=rerank_top_k)

    context = "\n\n".join([r["text"] for r in results])
    sources = [r["source"] for r in results]

    answer = (
        f"Based on the retrieved documentation, the answer to '{query}' is:\n\n"
        f"{results[0]['text'][:500]}"
    )

    return {
        "query": query,
        "answer": answer,
        "context": context,
        "sources": sources,
        "results": results,
    }