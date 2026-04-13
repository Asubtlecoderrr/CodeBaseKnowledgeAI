"""
rerank.py — Merge vector and BM25 results using Reciprocal Rank Fusion (RRF),
with an optional cross-encoder reranker for higher precision.

RRF formula: score(d) = Σ  1 / (k + rank_i(d))
where k=60 is the standard constant that prevents very high-ranked docs
from dominating. Works well without any training or extra API calls.

Cross-encoder (optional): feeds query+chunk pairs through a small BERT model
that scores relevance more precisely than embedding similarity. Slower but
more accurate for ambiguous queries. Enable via USE_RERANKER=True in config.
"""

from __future__ import annotations
from config import TOP_K, RRF_K, USE_RERANKER


class Reranker:
    def __init__(self):
        self._cross_encoder = None
        if USE_RERANKER:
            self._load_cross_encoder()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        vector_hits: list[dict],
        bm25_hits: list[dict],
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Merge and re-rank two result lists, return top_k final hits.

        Each hit dict must have at least: id, text, file, name.
        Additional keys (start_line, end_line, etc.) are preserved.
        """
        merged = _rrf_merge(vector_hits, bm25_hits, k=RRF_K)

        if USE_RERANKER and self._cross_encoder is not None:
            merged = self._cross_encode(query, merged)

        return merged[:top_k]

    # ------------------------------------------------------------------
    # Cross-encoder (optional)
    # ------------------------------------------------------------------

    def _load_cross_encoder(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            # Small, fast model. For better accuracy use: cross-encoder/ms-marco-MiniLM-L-12-v2
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("[rerank] cross-encoder loaded")
        except ImportError:
            print("[rerank] sentence-transformers not installed, skipping cross-encoder")

    def _cross_encode(self, query: str, hits: list[dict]) -> list[dict]:
        """Score each (query, chunk_text) pair and re-sort."""
        pairs = [(query, h["text"]) for h in hits]
        scores = self._cross_encoder.predict(pairs)
        for hit, score in zip(hits, scores):
            hit["ce_score"] = float(score)
        return sorted(hits, key=lambda h: h.get("ce_score", 0), reverse=True)


# ---------------------------------------------------------------------------
# RRF implementation
# ---------------------------------------------------------------------------

def _rrf_merge(
    list_a: list[dict],
    list_b: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion of two ranked lists.
    Documents are matched by their 'id' field.
    Returns merged list sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    docs:   dict[str, dict]  = {}

    for rank, hit in enumerate(list_a):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        docs[doc_id] = hit

    for rank, hit in enumerate(list_b):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        if doc_id not in docs:
            docs[doc_id] = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for doc_id, rrf_score in ranked:
        hit = dict(docs[doc_id])
        hit["rrf_score"] = round(rrf_score, 6)
        result.append(hit)

    return result