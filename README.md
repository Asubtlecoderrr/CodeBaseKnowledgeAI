# CodeBase Knowledge AI

AI-powered codebase Q&A using AST-aware chunking, hybrid vector + BM25 search, and a fully local LLM. Ask natural language questions about any codebase — runs entirely on your machine, no API keys required.

## How it works

1. **Index** — walks your repo, extracts functions and classes using tree-sitter (AST-aware), embeds them with a local sentence-transformer model, stores in ChromaDB
2. **Ask** — embeds your question locally, runs hybrid vector + BM25 search, reranks with RRF, sends top chunks to a local LLM (Ollama) for the answer

## Stack

| Layer | Tool |
|---|---|
| AST parsing | tree-sitter |
| Embeddings | all-MiniLM-L6-v2 (local, no API) |
| Vector DB | ChromaDB |
| Keyword search | BM25 (rank-bm25) |
| LLM | Ollama (llama3.2) |
| Server | FastAPI |

## Setup

### 1. Install Ollama

Download from https://ollama.com and install. Then:

```bash
ollama pull llama3.2
ollama serve
```

### 2. Install dependencies

```bash
cd codebase-qa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# .env is only needed if you switch to a cloud LLM
```

## Usage

### CLI

```bash
# Index your repo (run once, then on changes)
python cli.py index /path/to/your/repo

# Ask questions
python cli.py ask "where does authentication happen?"
python cli.py ask "how does the payment retry logic work?"
python cli.py ask "what does csn_generator do?"

# Interactive REPL
python cli.py ask

# Only re-index files changed since last commit
python cli.py index /path/to/your/repo --diff

# Show index stats
python cli.py stats

# Rebuild BM25 index from existing vectors
python cli.py reindex-bm25

# Re-upsert from cached vectors without re-embedding
python cli.py recover
```

### API server

```bash
uvicorn server:app --reload --port 8000
```

```
POST /ask           { "question": "where is auth?" }
POST /ask/stream    { "question": "..." }   # SSE stream
POST /index         { "repo_path": "/path/to/repo" }
GET  /stats
GET  /health
```

## File map

```
codebase-qa/
├── indexer/
│   ├── parser.py         AST chunking (tree-sitter)
│   ├── embedder.py       local sentence-transformer embeddings
│   └── ingest.py         walks repo, orchestrates indexing
├── retriever/
│   ├── search.py         vector + BM25 hybrid search
│   └── rerank.py         RRF merge + optional cross-encoder
├── qa/
│   ├── prompt.py         builds context + system prompt
│   └── claude.py         LLM generation (Ollama)
├── store/
│   └── chroma_client.py  ChromaDB wrapper
├── cli.py                CLI entry point
├── server.py             FastAPI server
├── config.py             all settings
└── data/                 vector DB and BM25 index (git-ignored)
```

## Supported languages

Python, TypeScript, JavaScript, Go, Java, Rust.
Other languages fall back to line-window chunking automatically.

## Switching to a cloud LLM

The `qa/claude.py` file is the only thing that needs to change. Swap Ollama for:
- **Gemini** — `google-generativeai` + `GOOGLE_API_KEY` in `.env`
- **Claude** — `anthropic` + `ANTHROPIC_API_KEY` in `.env`
- **OpenAI** — `openai` + `OPENAI_API_KEY` in `.env`
