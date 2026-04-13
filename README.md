# CodeBase Knowledge AI

AI-powered codebase Q&A using AST-aware chunking, hybrid search, and Claude.

## Setup

```bash
cd codebase-qa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY and GOOGLE_API_KEY
```

## Usage

### CLI

```bash
# Index your repo (run once, then on changes)
python cli.py index ../your-project/src

# Ask questions
python cli.py ask "where does authentication happen?"
python cli.py ask "how does the payment retry logic work?"

# Interactive REPL
python cli.py ask

# Only re-index changed files (uses git diff)
python cli.py index ../your-project/src --diff

# Show index stats
python cli.py stats
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
│   ├── parser.py       AST chunking (tree-sitter)
│   ├── embedder.py     text-embedding-004 embeddings
│   └── ingest.py       walks repo, orchestrates indexing
├── retriever/
│   ├── search.py       vector + BM25 hybrid search
│   └── rerank.py       RRF merge + optional cross-encoder
├── qa/
│   ├── prompt.py       builds context + system prompt
│   └── claude.py       calls Anthropic API
├── store/
│   └── chroma_client.py  ChromaDB wrapper
├── cli.py              CLI entry point
├── server.py           FastAPI server
├── config.py           all settings
└── data/               vector DB and BM25 index (git-ignored)
```

## Supported languages

Python, TypeScript, JavaScript, Go, Java, Rust.
Other languages fall back to line-window chunking automatically.
