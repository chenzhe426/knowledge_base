"""
Context assembly: build readable context text from retrieved chunks.
"""
from __future__ import annotations

from typing import Any

from app.services.common import normalize_whitespace

from .config import (
    CHAT_HISTORY_LIMIT,
    QA_MAX_CONTEXT_CHUNKS,
    _page_label,
    _truncate_text,
)


def assemble_context(chunks: list[dict[str, Any]], max_chunks: int = QA_MAX_CONTEXT_CHUNKS) -> str:
    """Build a readable context string from retrieved chunks."""
    if not chunks:
        return ""

    context_parts: list[str] = []

    for idx, chunk in enumerate(chunks[:max_chunks], start=1):
        title = chunk.get("title", "") or ""
        section_title = chunk.get("section_title", "") or ""
        section_path = chunk.get("section_path", "") or ""
        page_lbl = _page_label(chunk.get("page_start"), chunk.get("page_end"))
        chunk_text = normalize_whitespace(chunk.get("search_text") or chunk.get("chunk_text") or "")

        part = [
            f"[Source {idx}]",
            f"Document: {title or '-'}",
            f"Section: {section_title or section_path or '-'}",
            f"Page: {page_lbl}",
            "Content:",
            chunk_text,
        ]
        context_parts.append("\n".join(part))

    return "\n\n".join(context_parts)


def _format_history_for_prompt(history: list[dict[str, Any]], limit: int = CHAT_HISTORY_LIMIT) -> str:
    """Format chat history as a string for prompt injection."""
    if not history:
        return ""

    selected = history[-limit:]
    lines: list[str] = []

    for msg in selected:
        role = msg.get("role", "user")
        message = normalize_whitespace(msg.get("message", ""))
        if not message:
            continue

        role_label = "User" if role == "user" else "Assistant" if role == "assistant" else "System"
        lines.append(f"{role_label}: {message}")

    return "\n".join(lines).strip()
