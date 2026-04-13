"""
parser.py — AST-aware chunking using tree-sitter.

Walks a source file and extracts function and class definitions as
intact chunks. Never splits mid-body. Returns structured dicts with
metadata (file, name, type, line number) that flow through to the
vector DB and are surfaced in answers.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_go as tsgo
import tree_sitter_java as tsjava
import tree_sitter_rust as tsrust
from tree_sitter import Language, Parser, Node

from config import CHUNK_MAX_TOKENS


# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------

_LANG_MAP: dict[str, object] = {
    ".py": tspython,
    ".js": tsjavascript,
    ".jsx": tsjavascript,
    ".go": tsgo,
    ".java": tsjava,
    ".rs": tsrust,
}

# Node types that represent top-level callable/definable units per language
_CHUNK_NODE_TYPES: dict[str, set[str]] = {
    ".py":   {"function_definition", "class_definition", "decorated_definition"},
    ".js":   {"function_declaration", "class_declaration", "method_definition",
              "arrow_function", "function_expression"},
    ".jsx":  {"function_declaration", "class_declaration", "method_definition",
              "arrow_function", "function_expression"},
    ".go":   {"function_declaration", "method_declaration", "type_declaration"},
    ".java": {"method_declaration", "class_declaration", "interface_declaration"},
    ".rs":   {"function_item", "impl_item", "struct_item", "trait_item"},
}

# TypeScript is handled via the python tree-sitter-typescript package
try:
    import tree_sitter_typescript as tsts
    _LANG_MAP[".ts"] = tsts.language_typescript()
    _LANG_MAP[".tsx"] = tsts.language_tsx()
    _CHUNK_NODE_TYPES[".ts"]  = {"function_declaration", "class_declaration",
                                  "method_definition", "arrow_function"}
    _CHUNK_NODE_TYPES[".tsx"] = _CHUNK_NODE_TYPES[".ts"]
    _TS_PRELOADED = True
except Exception:
    _TS_PRELOADED = False


@dataclass
class Chunk:
    text: str                        # raw source of the chunk
    name: str                        # function / class name
    type: str                        # "function" | "class" | "method" etc.
    file: str                        # relative path to the source file
    start_line: int                  # 0-indexed line number
    end_line: int
    language: str                    # file extension e.g. ".py"
    enriched: str = field(default="")  # text with metadata header, used for embedding

    def __post_init__(self):
        if not self.enriched:
            self.enriched = (
                f"# File: {self.file}  |  {self.type}: {self.name}  "
                f"|  Lines {self.start_line+1}–{self.end_line+1}\n\n"
                + self.text
            )

    @property
    def id(self) -> str:
        return f"{self.file}:{self.name}:{self.start_line}"


# ---------------------------------------------------------------------------
# Parser factory
# ---------------------------------------------------------------------------

def _make_parser(ext: str) -> Optional[Parser]:
    lang_mod = _LANG_MAP.get(ext)
    if lang_mod is None:
        return None
    try:
        if isinstance(lang_mod, int):
            # already a raw language pointer (typescript preloaded path)
            lang = Language(lang_mod)
        else:
            lang = Language(lang_mod.language())
        p = Parser(lang)
        return p
    except Exception:
        return None


_parser_cache: dict[str, Optional[Parser]] = {}


def get_parser(ext: str) -> Optional[Parser]:
    if ext not in _parser_cache:
        _parser_cache[ext] = _make_parser(ext)
    return _parser_cache[ext]


# ---------------------------------------------------------------------------
# Name extraction helpers
# ---------------------------------------------------------------------------

def _get_node_name(node: Node, source: bytes) -> str:
    """Extract the identifier name from a definition node."""
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        # Go / Java / Rust often have a 'name' field
        if child.type in ("name", "field_identifier", "type_identifier"):
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return "anonymous"


def _normalize_type(node_type: str) -> str:
    if "function" in node_type or "method" in node_type or "arrow" in node_type:
        return "function"
    if "class" in node_type or "struct" in node_type or "impl" in node_type:
        return "class"
    if "interface" in node_type or "trait" in node_type:
        return "interface"
    return node_type


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_chunks(source_path: Path, repo_root: Path) -> list[Chunk]:
    """
    Parse a single source file and return a list of Chunk objects.
    Falls back to line-based windowing for unsupported languages.
    """
    ext = source_path.suffix.lower()
    rel_path = str(source_path.relative_to(repo_root))

    try:
        source_bytes = source_path.read_bytes()
        source_str = source_bytes.decode("utf-8", errors="replace")
    except OSError:
        return []

    parser = get_parser(ext)
    if parser is None:
        return _fallback_chunks(source_str, rel_path, ext)

    try:
        tree = parser.parse(source_bytes)
    except Exception:
        return _fallback_chunks(source_str, rel_path, ext)

    chunk_types = _CHUNK_NODE_TYPES.get(ext, set())
    chunks: list[Chunk] = []
    _walk(tree.root_node, source_bytes, source_str, rel_path, ext, chunk_types, chunks)

    # If AST found nothing (e.g. script-only file), fall back to windows
    if not chunks:
        return _fallback_chunks(source_str, rel_path, ext)

    return chunks


def _walk(
    node: Node,
    source_bytes: bytes,
    source_str: str,
    rel_path: str,
    ext: str,
    chunk_types: set[str],
    result: list[Chunk],
    depth: int = 0,
) -> None:
    """Recursive DFS over the AST."""
    if node.type in chunk_types:
        text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        name = _get_node_name(node, source_bytes)
        chunk = Chunk(
            text=text,
            name=name,
            type=_normalize_type(node.type),
            file=rel_path,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            language=ext,
        )
        # If chunk is too long, try to split it at inner method boundaries
        if _approx_tokens(text) > CHUNK_MAX_TOKENS and depth < 2:
            inner_chunks: list[Chunk] = []
            _walk(node, source_bytes, source_str, rel_path, ext,
                  chunk_types, inner_chunks, depth + 1)
            if inner_chunks:
                # Keep the outer signature as a stub chunk
                sig = _extract_signature(text)
                stub = Chunk(
                    text=sig,
                    name=name + " (signature)",
                    type=chunk.type,
                    file=rel_path,
                    start_line=chunk.start_line,
                    end_line=chunk.start_line + sig.count("\n"),
                    language=ext,
                )
                result.append(stub)
                result.extend(inner_chunks)
                return
        result.append(chunk)
        return  # Don't recurse into already-captured nodes

    for child in node.children:
        _walk(child, source_bytes, source_str, rel_path, ext,
              chunk_types, result, depth)


def _extract_signature(text: str, max_lines: int = 5) -> str:
    """Return just the first few lines of a definition (the signature)."""
    lines = text.splitlines()
    return "\n".join(lines[:max_lines]) + "\n    ..."


def _approx_tokens(text: str) -> int:
    """Very rough token estimate: 1 token ≈ 4 chars."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Fallback: line-window chunking for unsupported languages
# ---------------------------------------------------------------------------

def _fallback_chunks(
    source: str,
    rel_path: str,
    ext: str,
    window: int = 40,
    stride: int = 20,
) -> list[Chunk]:
    lines = source.splitlines()
    chunks = []
    i = 0
    while i < len(lines):
        block = lines[i : i + window]
        text = "\n".join(block)
        chunks.append(Chunk(
            text=text,
            name=f"lines_{i+1}_{i+len(block)}",
            type="block",
            file=rel_path,
            start_line=i,
            end_line=i + len(block) - 1,
            language=ext,
        ))
        i += stride
    return chunks