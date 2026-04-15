from typing import List, Tuple

from core.models import reranker


def rerank(query: str, docs: List[str], top_k: int = 10) -> Tuple[List[str], List[float], float]:
    """
    Rerank a list of document strings using the bge-reranker-v2-m3 cross-encoder.
    Returns (top_k most relevant documents, their scores, best relevance score).
    Scores are raw logits: >0 = relevant, < -4 = likely irrelevant.
    """
    pairs = [[query, d] for d in docs]
    raw = reranker.compute_score(pairs)
    scores = [float(raw)] if not hasattr(raw, '__len__') else list(raw)

    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_docs = [docs[i] for i in sorted_idx[:top_k]]
    top_scores = [scores[i] for i in sorted_idx[:top_k]]
    best_score = top_scores[0]
    return top_docs, top_scores, best_score
