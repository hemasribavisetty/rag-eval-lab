from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = CrossEncoder(RERANK_MODEL)

def rerank(query: str, candidates: list[dict], text_key: str = "text"):
    pairs = [(query, c[text_key]) for c in candidates]
    scores = _cross_encoder.predict(pairs)

    reranked = []
    for c, s in zip(candidates, scores):
        item = dict(c)
        item["rerank_score"] = float(s)
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked