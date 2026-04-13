"""
server.py — FastAPI server for the codebase Q&A tool.

Endpoints:
    POST /ask           { question: str }  → { answer: str, sources: list }
    POST /ask/stream    { question: str }  → SSE stream of answer tokens
    POST /index         { repo_path: str, diff: bool }
    GET  /stats                            → { chunk_count: int }

Run with:
    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from indexer.ingest import Ingester
from retriever.search import HybridSearcher
from retriever.rerank import Reranker
from qa.prompt import build_messages, format_sources
from qa.claude import ClaudeQA
from store.chroma_client import ChromaStore
from config import TOP_K


# ---------------------------------------------------------------------------
# Singletons (initialised once at startup)
# ---------------------------------------------------------------------------

_searcher: HybridSearcher | None = None
_reranker: Reranker | None = None
_qa:       ClaudeQA | None = None
_store:    ChromaStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _searcher, _reranker, _qa, _store
    _searcher = HybridSearcher()
    _reranker = Reranker()
    _qa       = ClaudeQA()
    _store    = ChromaStore()
    yield


app = FastAPI(
    title="Codebase Q&A API",
    description="AI-powered codebase question answering backed by Claude",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]

class IndexRequest(BaseModel):
    repo_path: str
    diff: bool = False

class IndexResponse(BaseModel):
    files: int
    chunks: int
    vectors_upserted: int
    elapsed_s: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Ask a question. Returns the full answer once complete."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    vector_hits, bm25_hits = _searcher.search(req.question)
    hits = _reranker.rerank(req.question, vector_hits, bm25_hits, top_k=req.top_k)

    if not hits:
        raise HTTPException(status_code=404, detail="No relevant code found in index")

    system, messages = build_messages(req.question, hits)
    answer = _qa.answer(system, messages)
    sources = format_sources(hits)

    return AskResponse(answer=answer, sources=sources)


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """
    Ask a question. Returns Server-Sent Events stream of answer tokens.
    Each event is a plain text chunk. The stream ends with a [DONE] event.

    Client example (JS):
        const es = new EventSource('/ask/stream', { method: 'POST', body: JSON.stringify({question}) })
        es.onmessage = e => process.stdout.write(e.data)
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    vector_hits, bm25_hits = _searcher.search(req.question)
    hits = _reranker.rerank(req.question, vector_hits, bm25_hits, top_k=req.top_k)

    if not hits:
        raise HTTPException(status_code=404, detail="No relevant code found in index")

    system, messages = build_messages(req.question, hits)

    def event_stream():
        for token in _qa.ask_stream(system, messages):
            yield f"data: {token}\n\n"
        sources = format_sources(hits)
        import json
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/index", response_model=IndexResponse)
async def index(req: IndexRequest) -> IndexResponse:
    """Index (or re-index) a repository. Rebuilds BM25 index after."""
    try:
        ingester = Ingester()
        stats = ingester.run(req.repo_path, mode="diff" if req.diff else "full")
        _searcher.rebuild_bm25_index()
        return IndexResponse(**stats)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/stats")
async def stats() -> dict:
    """Return index statistics."""
    return {"chunk_count": _store.count()}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}