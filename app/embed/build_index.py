# app/embed/build_index.py
import json
import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

EMBED_MODEL = "all-MiniLM-L6-v2"

def load_chunks(chunk_file):
    chunks = []
    metas = []
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(obj["text"])
            metas.append({"doc_id": obj["doc_id"], "chunk_id": obj["chunk_id"], "source": obj["source_file"]})
    return chunks, metas

def build(chunk_file: str, index_file: str, meta_file: str):
    chunks, metas = load_chunks(chunk_file)
    print(f"[INFO] Loaded {len(chunks)} chunks from {chunk_file}")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True, batch_size=32)
    dim = embeddings.shape[1]
    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    Path(index_file).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_file)
    print(f"[DONE] Wrote FAISS index to {index_file}")

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Wrote metadata to {meta_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_file", default="experiments/chunks.jsonl")
    parser.add_argument("--index_file", default="experiments/faiss.index")
    parser.add_argument("--meta_file", default="experiments/chunks_meta.json")
    args = parser.parse_args()

    build(args.chunk_file, args.index_file, args.meta_file)
