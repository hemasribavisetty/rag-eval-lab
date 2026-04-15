import json
import faiss
from sentence_transformers import SentenceTransformer
from app.retrieve.reranker import rerank

EMBED_MODEL = "all-MiniLM-L6-v2"
_model = SentenceTransformer(EMBED_MODEL)

class Retriever:
    def __init__(self, index_file: str, meta_file: str, chunk_file: str):
        self.index = faiss.read_index(index_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metas = json.load(f)
        self.chunks = []
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line)["text"])

    def search(self, query: str, top_k: int = 5, rerank_top_k: int = 0):
        q_emb = _model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        D, I = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({
                "score": float(score),
                "source": self.metas[idx]["source"],
                "doc_id": self.metas[idx]["doc_id"],
                "chunk_id": self.metas[idx]["chunk_id"],
                "text": self.chunks[idx][:800]
            })

        if rerank_top_k and rerank_top_k > 1:
            head = rerank(query, results[:rerank_top_k])
            tail = results[rerank_top_k:]
            results = head + tail

        return results
