"""
search.py — Hybrid vector + BM25 keyword search.

Why hybrid?
  - Vector search finds semantically similar code even with different naming.
  - BM25 catches exact symbol names, error codes, and identifiers that
    embeddings can blur (e.g. "AuthMiddleware" vs "auth_middleware").
  - Combining both with RRF gives the best of both worlds.

The BM25 index is built lazily from the ChromaDB store and cached to disk.
On re-index it is rebuilt automatically.
"""

from __future__ import annotations
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from store.chroma_client import ChromaStore
from indexer.embedder import Embedder
from config import VECTOR_TOP_K, BM25_TOP_K, BM25_INDEX_PATH


class HybridSearcher:
    def __init__(self):
        self._store = ChromaStore()
        self._embedder = Embedder()
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []   # parallel list of chunk ids for the BM25 index

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = VECTOR_TOP_K) -> tuple[list[dict], list[dict]]:
        """
        Run both searches.

        Returns:
            (vector_results, bm25_results) — each is a list of hit dicts,
            ordered by score descending.
            The caller (rerank.py) merges them with RRF.
        """
        query_vector = self._embedder.embed_query(query)
        vector_hits  = self._store.query(query_vector, top_k=top_k)
        bm25_hits    = self._bm25_search(query, top_k=BM25_TOP_K)
        return vector_hits, bm25_hits

    def rebuild_bm25_index(self) -> None:
        """Rebuild the BM25 index from all chunks in the store. Saves to disk."""
        print("[search] rebuilding BM25 index …")
        all_chunks = self._store.get_all_texts()
        self._bm25_ids = [c["id"] for c in all_chunks]
        tokenized = [_tokenize(c["text"]) for c in all_chunks]
        self._bm25 = BM25Okapi(tokenized)

        BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": self._bm25, "ids": self._bm25_ids}, f)
        print(f"[search] BM25 index saved ({len(self._bm25_ids)} docs)")

    # ------------------------------------------------------------------
    # BM25 internals
    # ------------------------------------------------------------------

    def _load_bm25(self) -> None:
        if self._bm25 is not None:
            return
        if BM25_INDEX_PATH.exists():
            with open(BM25_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._bm25_ids = data["ids"]
        else:
            self.rebuild_bm25_index()

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        self._load_bm25()
        if not self._bm25_ids:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair (score, id) and take top_k
        ranked = sorted(
            zip(scores, self._bm25_ids),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        # Fetch full chunk metadata for matched ids
        hit_ids = {chunk_id: score for score, chunk_id in ranked if score > 0}
        if not hit_ids:
            return []

        all_chunks = self._store.get_all_texts()
        results = []
        for chunk in all_chunks:
            if chunk["id"] in hit_ids:
                results.append({
                    "id":    chunk["id"],
                    "file":  chunk["file"],
                    "name":  chunk["name"],
                    "text":  chunk["text"],
                    "score": float(hit_ids[chunk["id"]]),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """
    Split code into tokens for BM25.
    Splits on whitespace and common code punctuation.
    Also splits camelCase and snake_case so "getUserById" matches "getUser".
    """
    import re
    # Insert spaces before uppercase letters in camelCase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Split on non-alphanumeric chars
    tokens = re.split(r"[^a-zA-Z0-9_]+", text)
    # Split snake_case
    expanded = []
    for tok in tokens:
        expanded.extend(tok.split("_"))
    return [t.lower() for t in expanded if len(t) > 1]