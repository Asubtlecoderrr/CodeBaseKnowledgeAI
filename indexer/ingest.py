"""
ingest.py — Walk a repo, extract AST chunks, embed them, store in ChromaDB.

Two modes:
  full   — re-index everything (first run or full refresh)
  diff   — only re-index files changed since last run (uses git diff or mtime)

Usage:
    from indexer.ingest import Ingester
    ingester = Ingester()
    ingester.run("/path/to/repo", mode="full")
"""

from __future__ import annotations
from pathlib import Path
import subprocess
import time
import pickle
from pathlib import Path
from indexer.parser import extract_chunks, Chunk
from indexer.embedder import Embedder
from store.chroma_client import ChromaStore
from config import EXTENSIONS


class Ingester:
    def __init__(self):
        self._embedder = Embedder()
        self._store = ChromaStore()


    def run(self, repo_path: str | Path, mode: str = "full") -> dict:
        """
        Index a repository.

        Args:
            repo_path: Absolute path to the repo root.
            mode:      "full" — index all files.
                       "diff" — only files changed since last git commit.

        Returns:
            Stats dict: { files, chunks, vectors_upserted, elapsed_s }
        """
        repo = Path(repo_path).resolve()
        if not repo.exists():
            raise FileNotFoundError(f"Repo path does not exist: {repo}")

        t0 = time.time()
        files = self._collect_files(repo, mode)
        print(f"[ingest] found {len(files)} files to index  (mode={mode})")

        all_chunks: list[Chunk] = []
        for f in files:
            chunks = extract_chunks(f, repo)
            all_chunks.extend(chunks)

        print(f"[ingest] extracted {len(all_chunks)} chunks")

        if not all_chunks:
            return {"files": len(files), "chunks": 0, "vectors_upserted": 0,
                    "elapsed_s": round(time.time() - t0, 2)}

        print(f"[ingest] embedding {len(all_chunks)} chunks …")
        vectors = self._embedder.embed(all_chunks)
        
        cache_path = Path("data/vectors_cache.pkl")
        cache_path.parent.mkdir(exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"chunks": all_chunks, "vectors": vectors}, f)
        print("[ingest] vectors cached to disk ✓")

        print(f"[ingest] upserting to vector DB …")
        self._store.upsert(all_chunks, vectors)
        print(f"[ingest] upserting to vector DB …")
        self._store.upsert(all_chunks, vectors)

        elapsed = round(time.time() - t0, 2)
        stats = {
            "files": len(files),
            "chunks": len(all_chunks),
            "vectors_upserted": len(vectors),
            "elapsed_s": elapsed,
        }
        print(f"[ingest] done in {elapsed}s  {stats}")
        return stats

    def delete_file(self, repo_path: str | Path, file_path: str | Path) -> int:
        """Remove all chunks belonging to a specific file from the store."""
        repo = Path(repo_path).resolve()
        rel = str(Path(file_path).relative_to(repo))
        return self._store.delete_by_file(rel)

    # ------------------------------------------------------------------
    # File collection
    # ------------------------------------------------------------------

    def _collect_files(self, repo: Path, mode: str) -> list[Path]:
        if mode == "diff":
            changed = self._git_changed_files(repo)
            if changed is not None:
                return [
                    repo / f for f in changed
                    if Path(f).suffix.lower() in EXTENSIONS
                    and (repo / f).exists()
                ]
            # git not available — fall through to full
            print("[ingest] git diff unavailable, falling back to full index")

        return [
            f for f in repo.rglob("*")
            if f.is_file()
            and f.suffix.lower() in EXTENSIONS
            and not _is_ignored(f)
        ]

    @staticmethod
    def _git_changed_files(repo: Path) -> list[str] | None:
        """Return list of files changed in the last commit, or None on error."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IGNORE_DIRS = {
    ".git", ".github", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "coverage", ".mypy_cache",
    ".pytest_cache", "codebase-qa", "__init__.py", 
}


def _is_ignored(path: Path) -> bool:
    for part in path.parts:
        if part in _IGNORE_DIRS:
            return True
        if part.startswith("."):
            return True
    return False