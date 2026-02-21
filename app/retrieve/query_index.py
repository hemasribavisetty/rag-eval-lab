# app/retrieve/query_index.py
import json
import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE = "experiments/faiss.index"
META_FILE = "experiments/chunks_meta.json"
CHUNK_FILE = "experiments/chunks.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Load once (module-level singletons)
_model = SentenceTransformer(EMBED_MODEL)
_index = faiss.read_index(INDEX_FILE)

with open(META_FILE, "r", encoding="utf-8") as f:
    _metas = json.load(f)

_chunks = []
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    for line in f:
        _chunks.append(json.loads(line)["text"])

def search(query: str, top_k: int = 5):
    q_emb = _model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    D, I = _index.search(q_emb, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        results.append({
            "score": float(score),
            "source": _metas[idx]["source"],
            "doc_id": _metas[idx]["doc_id"],
            "chunk_id": _metas[idx]["chunk_id"],
            "text": _chunks[idx][:800]
        })
    return results

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How do FastAPI dependencies work?"
    res = search(q, top_k=5)
    print(f"\nQuery: {q}\n")
    for r in res:
        print(f"score={r['score']:.4f}  source={r['source']}  (doc={r['doc_id']}, chunk={r['chunk_id']})")
        print(r["text"])
        print("-" * 80)
