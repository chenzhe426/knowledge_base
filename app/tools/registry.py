from __future__ import annotations

from typing import Callable, Dict

from app.tools.kb_history_tools import kb_create_chat_session, kb_get_chat_history
from app.tools.kb_import_tools import kb_import_file, kb_import_folder
from app.tools.kb_index_tools import kb_index_document
from app.tools.kb_search_tools import kb_search_knowledge_base
from app.tools.kb_summary_tools import kb_summarize_document


TOOL_REGISTRY: Dict[str, Callable] = {
    "kb_import_file": kb_import_file,
    "kb_import_folder": kb_import_folder,
    "kb_index_document": kb_index_document,
    "kb_summarize_document": kb_summarize_document,
    "kb_create_chat_session": kb_create_chat_session,
    "kb_get_chat_history": kb_get_chat_history,
    "kb_search_knowledge_base": kb_search_knowledge_base,
}


def get_tool(name: str) -> Callable:
    try:
        return TOOL_REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"tool not found: {name}") from e


def list_tools() -> list[str]:
    return list(TOOL_REGISTRY.keys())