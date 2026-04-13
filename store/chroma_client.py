"""
chroma_client.py — ChromaDB wrapper for storing and querying code chunk vectors.

Keeps the rest of the codebase DB-agnostic. To swap to pgvector or Pinecone,
replace this file only — the interface (upsert / query / delete_by_file) stays the same.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import json

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, COLLECTION_NAME, VECTOR_TOP_K

if TYPE_CHECKING:
    from indexer.parser import Chunk


class ChromaStore:
    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},   # cosine similarity for code embeddings
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, chunks: list["Chunk"], vectors: list[list[float]]) -> None:
        if not chunks:
            return

        # Deduplicate by id — keep last occurrence
        seen = {}
        for chunk, vector in zip(chunks, vectors):
            seen[chunk.id] = (chunk, vector)
        
        chunks  = [c for c, v in seen.values()]
        vectors = [v for c, v in seen.values()]

        ids        = [c.id for c in chunks]
        documents  = [c.enriched for c in chunks]
        metadatas  = [
            {
                "file":       c.file,
                "name":       c.name,
                "type":       c.type,
                "start_line": c.start_line,
                "end_line":   c.end_line,
                "language":   c.language,
                "raw_text":   c.text,
            }
            for c in chunks
        ]

        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self._col.upsert(
                ids=ids[i:i+batch_size],
                embeddings=vectors[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )
            
    def delete_by_file(self, rel_path: str) -> int:
        """Delete all chunks belonging to a file. Returns count deleted."""
        results = self._col.get(where={"file": rel_path}, include=[])
        ids = results.get("ids", [])
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        vector: list[float],
        top_k: int = VECTOR_TOP_K,
        file_filter: str | None = None,
    ) -> list[dict]:
        """
        Return top_k most similar chunks as dicts with keys:
        id, file, name, type, start_line, end_line, language, text, score
        """
        where = {"file": file_filter} if file_filter else None

        results = self._col.query(
            query_embeddings=[vector],
            n_results=min(top_k, self._col.count() or 1),
            include=["metadatas", "documents", "distances"],
            where=where,
        )

        hits = []
        ids       = results["ids"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, meta, dist in zip(ids, metas, distances):
            hits.append({
                "id":         chunk_id,
                "file":       meta["file"],
                "name":       meta["name"],
                "type":       meta["type"],
                "start_line": meta["start_line"],
                "end_line":   meta["end_line"],
                "language":   meta["language"],
                "text":       meta.get("raw_text", ""),
                "score":      1.0 - dist,   # cosine distance → similarity
            })
        return hits

    def get_all_texts(self) -> list[dict]:
        """Return all stored chunks (for BM25 index rebuild)."""
        results = self._col.get(include=["metadatas"])
        hits = []
        for chunk_id, meta in zip(results["ids"], results["metadatas"]):
            hits.append({
                "id":   chunk_id,
                "file": meta["file"],
                "name": meta["name"],
                "text": meta.get("raw_text", ""),
            })
        return hits

    def count(self) -> int:
        return self._col.count()