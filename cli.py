"""
cli.py — Command-line interface for the codebase Q&A tool.

Commands:
    python cli.py index <repo_path>          # full index
    python cli.py index <repo_path> --diff   # only changed files
    python cli.py ask "<question>"           # ask a question
    python cli.py ask                        # interactive REPL mode
    python cli.py stats                      # show index stats
    python cli.py reindex-bm25               # rebuild BM25 index from existing vectors
"""

import sys
import argparse

from indexer.ingest import Ingester
from retriever.search import HybridSearcher
from retriever.rerank import Reranker
from qa.prompt import build_messages, format_sources
from qa.claude import ClaudeQA
from store.chroma_client import ChromaStore
from config import TOP_K


def cmd_index(args: argparse.Namespace) -> None:
    mode = "diff" if args.diff else "full"
    ingester = Ingester()
    stats = ingester.run(args.repo_path, mode=mode)
    print(f"\nIndexed {stats['chunks']} chunks from {stats['files']} files in {stats['elapsed_s']}s")

    # Rebuild BM25 index after (re)indexing
    print("Rebuilding BM25 index …")
    HybridSearcher().rebuild_bm25_index()
    print("Done.")


def cmd_ask(args: argparse.Namespace) -> None:
    searcher = HybridSearcher()
    reranker = Reranker()
    qa       = ClaudeQA()

    if args.question:
        _answer_once(args.question, searcher, reranker, qa)
    else:
        _repl(searcher, reranker, qa)


def _answer_once(question: str, searcher, reranker, qa) -> None:
    print(f"\nSearching …")
    vector_hits, bm25_hits = searcher.search(question)
    hits = reranker.rerank(question, vector_hits, bm25_hits, top_k=TOP_K)

    if not hits:
        print("No relevant code found in the index. Have you run `python cli.py index <repo>`?")
        return

    system, messages = build_messages(question, hits)

    print(f"\nAnswer (citing {len(hits)} chunks):\n")
    print("─" * 60)
    for token in qa.ask_stream(system, messages):
        print(token, end="", flush=True)
    print("\n" + "─" * 60)

    sources = format_sources(hits)
    print("\nSources:")
    for s in sources:
        print(f"  {s['file']} → {s['name']} (line {s['start_line']})")


def _repl(searcher, reranker, qa) -> None:
    print("\nCodebase Q&A — interactive mode. Type 'quit' or Ctrl-C to exit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        _answer_once(question, searcher, reranker, qa)
        print()


def cmd_stats(args: argparse.Namespace) -> None:
    store = ChromaStore()
    count = store.count()
    print(f"Vector DB: {count} chunks indexed")


def cmd_reindex_bm25(args: argparse.Namespace) -> None:
    HybridSearcher().rebuild_bm25_index()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="codebase-qa",
        description="AI-powered codebase Q&A",
    )
    sub = parser.add_subparsers(dest="command")

    # index
    p_index = sub.add_parser("index", help="Index a repository")
    p_index.add_argument("repo_path", help="Path to the repository root")
    p_index.add_argument("--diff", action="store_true",
                         help="Only re-index files changed since last git commit")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question about the codebase")
    p_ask.add_argument("question", nargs="?", default=None,
                       help="Question to ask (omit for interactive REPL)")

    # stats
    sub.add_parser("stats", help="Show index statistics")

    # reindex-bm25
    sub.add_parser("reindex-bm25", help="Rebuild BM25 index from existing vector store")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "reindex-bm25":
        cmd_reindex_bm25(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()