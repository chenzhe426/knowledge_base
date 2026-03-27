"""
Backward-compatible facade for QA.

Actual logic has been moved to app/qa/ submodules.
This file re-exports the public API for existing import points.
"""
from __future__ import annotations

# Import from submodules (submodules import from app.retrieval.* which is circular-import-safe)
from app.qa.pipeline import answer_question, rewrite_query_with_history
from app.qa.session import get_chat_history, summarize_document
from app.qa.context import assemble_context

__all__ = [
    "answer_question",
    "assemble_context",
    "get_chat_history",
    "rewrite_query_with_history",
    "summarize_document",
]
