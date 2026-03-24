from langchain.tools import tool

from app.tools import (
    kb_get_chat_history,
    kb_search_knowledge_base,
    kb_summarize_document,
)


@tool
def search_knowledge_base(query: str, top_k: int = 5, include_full_text: bool = False) -> dict:
    """在知识库中搜索相关内容。适用于事实问答、文档检索、概念解释、对比分析。"""
    return kb_search_knowledge_base(
        {
            "query": query,
            "top_k": top_k,
            "include_full_text": include_full_text,
        }
    )


@tool
def summarize_document(document_id: int) -> dict:
    """根据 document_id 对指定文档做摘要。"""
    return kb_summarize_document(
        {
            "document_id": document_id,
        }
    )


@tool
def get_chat_history(session_id: str, limit: int = 20) -> dict:
    """根据 session_id 获取历史对话消息。"""
    return kb_get_chat_history(
        {
            "session_id": session_id,
            "limit": limit,
        }
    )


TOOLS = [
    search_knowledge_base,
    summarize_document,
]