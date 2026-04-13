"""
qa/prompt.py — Build the system prompt and user message sent to Claude.

The context block contains retrieved code chunks, each labelled with
file path and function name so Claude can cite them precisely.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert software engineer assistant with deep knowledge of this codebase.

When answering questions:
1. Base your answer ONLY on the code context provided below.
2. Always cite the specific file and function/class name (e.g. `auth/middleware.py → verify_token()`).
3. If the context doesn't contain enough information to answer fully, say so explicitly — do not guess.
4. When relevant, include short code snippets from the context to support your answer.
5. If multiple files are relevant, mention all of them.
6. Keep answers concise and actionable.
"""


def build_messages(
    question: str,
    hits: list[dict],
) -> tuple[str, list[dict]]:
    """
    Build (system_prompt, messages) ready to pass to the Anthropic API.

    Args:
        question: The user's natural-language question.
        hits:     Reranked list of chunk dicts from the retriever.

    Returns:
        (system_prompt_str, messages_list)
    """
    context = _format_context(hits)
    user_message = f"{context}\n\n---\n\nQuestion: {question}"
    messages = [{"role": "user", "content": user_message}]
    return SYSTEM_PROMPT, messages


def build_follow_up_messages(
    history: list[dict],
    question: str,
    hits: list[dict],
) -> tuple[str, list[dict]]:
    """
    Build messages for a follow-up question in a multi-turn session.
    Injects fresh context from the new retrieval into the latest user turn.
    """
    context = _format_context(hits)
    new_user = f"Additional context for this follow-up:\n{context}\n\n---\n\nFollow-up: {question}"
    messages = list(history) + [{"role": "user", "content": new_user}]
    return SYSTEM_PROMPT, messages


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_context(hits: list[dict]) -> str:
    if not hits:
        return "No relevant code found in the index."

    parts = ["### Retrieved code context\n"]
    for i, hit in enumerate(hits, 1):
        file_loc = f"{hit['file']} → {hit.get('name', 'unknown')} (line {hit.get('start_line', '?')+1})"
        parts.append(f"**[{i}] {file_loc}**\n```\n{hit['text']}\n```")

    return "\n\n".join(parts)


def format_sources(hits: list[dict]) -> list[dict]:
    """
    Return a clean list of source references for the API response.
    """
    seen = set()
    sources = []
    for hit in hits:
        key = f"{hit['file']}:{hit.get('name', '')}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file":       hit["file"],
                "name":       hit.get("name", ""),
                "type":       hit.get("type", ""),
                "start_line": hit.get("start_line", 0) + 1,  # 1-indexed for display
                "end_line":   hit.get("end_line", 0) + 1,
            })
    return sources