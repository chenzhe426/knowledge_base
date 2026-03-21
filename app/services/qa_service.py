from app.config import DEFAULT_TOP_K
from app.db import get_document_by_id
from app.services.common import safe_get, truncate
from app.services.llm_service import chat_completion, summarize_text
from app.services.retrieval_service import retrieve_chunks


def assemble_context(retrieved_chunks: list[dict]) -> str:
    context_parts = []

    for idx, item in enumerate(retrieved_chunks, start=1):
        title = item.get("doc_title") or "Untitled"
        section_path = item.get("section_path") or "-"
        page_start = item.get("page_start")
        page_end = item.get("page_end")
        page_info = "-"
        if page_start is not None and page_end is not None:
            page_info = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        elif page_start is not None:
            page_info = str(page_start)
        elif page_end is not None:
            page_info = str(page_end)

        chunk_text = item.get("chunk_text", "")

        context_parts.append(
            f"[片段{idx}]\n"
            f"文档标题: {title}\n"
            f"章节路径: {section_path}\n"
            f"页码: {page_info}\n"
            f"内容:\n{chunk_text}"
        )

    return "\n\n".join(context_parts).strip()


def answer_question(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    retrieved = retrieve_chunks(question, top_k=top_k)
    if not retrieved:
        return {
            "question": question,
            "answer": "没有检索到相关内容，无法基于知识库回答这个问题。",
            "sources": [],
        }

    context = assemble_context(retrieved)

    system = (
        "你是一个基于知识库回答问题的助手。\n"
        "要求：\n"
        "1. 只能依据提供的上下文回答。\n"
        "2. 如果上下文不足，请明确说明“根据当前检索内容无法确定”。\n"
        "3. 优先给出直接答案，再给出简要依据。\n"
        "4. 不要编造原文中没有的信息。"
    )

    prompt = (
        f"用户问题：{question}\n\n"
        f"检索上下文：\n{context}\n\n"
        "请根据以上内容回答问题。"
    )

    answer = chat_completion(prompt=prompt, system=system)

    sources = []
    for item in retrieved:
        sources.append(
            {
                "document_id": item.get("document_id"),
                "chunk_id": item.get("chunk_id"),
                "chunk_index": item.get("chunk_index"),
                "score": item.get("score"),
                "doc_title": item.get("doc_title"),
                "section_path": item.get("section_path"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "chunk_type": item.get("chunk_type"),
                "preview": truncate(item.get("chunk_text", ""), 160),
            }
        )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


def summarize_document(doc_id: int) -> dict:
    row = get_document_by_id(doc_id)
    if not row:
        raise ValueError("document not found")

    title = safe_get(row, "title", "") or ""
    content = safe_get(row, "content", "") or ""

    summary = summarize_text(content)
    return {
        "document_id": doc_id,
        "title": title,
        "summary": summary,
    }