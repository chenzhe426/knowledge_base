import json
import re
import uuid
from typing import Any

import app.config as config
from app.db import (
    create_chat_session,
    get_chat_messages,
    get_chat_session,
    get_document_by_id,
    insert_chat_message,
    update_chat_session,
)
from app.services.common import normalize_whitespace, safe_get, to_float
from app.services.llm_service import chat_completion, summarize_text
from app.services.retrieval_service import retrieve_chunks


def _cfg(name: str, default: Any):
    return getattr(config, name, default)


CHAT_HISTORY_LIMIT = int(_cfg("CHAT_HISTORY_LIMIT", 6))
CHAT_SUMMARY_TRIGGER_TURNS = int(_cfg("CHAT_SUMMARY_TRIGGER_TURNS", 12))
QA_MAX_CONTEXT_CHUNKS = int(_cfg("QA_MAX_CONTEXT_CHUNKS", 6))
QA_STRUCTURED_ENABLE = bool(_cfg("QA_STRUCTURED_ENABLE", True))
QA_HIGHLIGHT_ENABLE = bool(_cfg("QA_HIGHLIGHT_ENABLE", True))


def _new_session_id() -> str:
    return uuid.uuid4().hex


def _truncate_text(text: str, max_len: int = 220) -> str:
    text = normalize_whitespace(text or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _page_label(page_start: Any, page_end: Any) -> str:
    if page_start is None and page_end is None:
        return "-"
    if page_start is not None and page_end is not None:
        if page_start == page_end:
            return str(page_start)
        return f"{page_start}-{page_end}"
    return str(page_start if page_start is not None else page_end)


def _normalize_chunk_source(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    统一 source 输出结构。
    该结构应可被 AnswerSource 直接接收。
    """
    return {
        "chunk_id": chunk.get("chunk_id"),
        "document_id": chunk.get("document_id"),
        "doc_title": chunk.get("doc_title", "") or "",
        "section_title": chunk.get("section_title", "") or "",
        "section_path": chunk.get("section_path", "") or "",
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "quote": chunk.get("chunk_text", "") or "",
        "score": to_float(chunk.get("score")),
        "highlight_spans": [],
    }


def _build_highlight_spans(text: str, terms: list[str]) -> list[dict[str, Any]]:
    text = text or ""
    if not text or not terms:
        return []

    spans: list[tuple[int, int]] = []

    for term in terms:
        term = normalize_whitespace(term)
        if not term:
            continue

        try:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
        except Exception:
            continue

        for match in pattern.finditer(text):
            start, end = match.span()
            if start == end:
                continue
            spans.append((start, end))

    if not spans:
        return []

    spans.sort(key=lambda x: (x[0], x[1]))
    merged: list[list[int]] = []

    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return [
        {
            "start": start,
            "end": end,
            "text": text[start:end],
        }
        for start, end in merged
    ]


def _extract_query_terms(
    question: str,
    rewritten_query: str | None,
    retrieved_chunks: list[dict[str, Any]],
) -> list[str]:
    terms = set()

    for raw in [question, rewritten_query or ""]:
        raw = normalize_whitespace(raw)
        if not raw:
            continue

        for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_./:-]{1,}", raw):
            terms.add(token)

    for chunk in retrieved_chunks:
        for term, count in (chunk.get("term_hits") or {}).items():
            if count:
                terms.add(term)

    return sorted(terms, key=lambda x: (-len(x), x))


def _normalize_retrieved_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    统一 retrieved_chunks 输出结构。
    该结构应可被 ChunkResult 直接接收。
    """
    return {
        "chunk_id": chunk.get("chunk_id"),
        "document_id": chunk.get("document_id"),
        "chunk_index": chunk.get("chunk_index"),
        "score": to_float(chunk.get("score")),
        "embedding_score": to_float(chunk.get("embedding_score")),
        "keyword_score": to_float(chunk.get("keyword_score")),
        "bm25_score": to_float(chunk.get("bm25_score")),
        "title_match_score": to_float(chunk.get("title_match_score")),
        "section_match_score": to_float(chunk.get("section_match_score")),
        "coverage_score": to_float(chunk.get("coverage_score")),
        "matched_term_count": chunk.get("matched_term_count"),
        "doc_title": chunk.get("doc_title", "") or "",
        "section_title": chunk.get("section_title", "") or "",
        "section_path": chunk.get("section_path", "") or "",
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "chunk_type": chunk.get("chunk_type"),
        "chunk_text": chunk.get("chunk_text", "") or "",
        "term_hits": chunk.get("term_hits") or {},
        "term_hit_detail": chunk.get("term_hit_detail") or {},
        "is_neighbor": bool(chunk.get("is_neighbor", False)),
    }


def assemble_context(chunks: list[dict[str, Any]], max_chunks: int = QA_MAX_CONTEXT_CHUNKS) -> str:
    if not chunks:
        return ""

    context_parts: list[str] = []

    for idx, chunk in enumerate(chunks[:max_chunks], start=1):
        doc_title = chunk.get("doc_title", "") or ""
        section_title = chunk.get("section_title", "") or ""
        section_path = chunk.get("section_path", "") or ""
        page_label = _page_label(chunk.get("page_start"), chunk.get("page_end"))
        chunk_text = normalize_whitespace(chunk.get("search_text") or chunk.get("chunk_text") or "")

        part = [
            f"[来源 {idx}]",
            f"文档：{doc_title or '-'}",
            f"章节：{section_title or section_path or '-'}",
            f"页码：{page_label}",
            "内容：",
            chunk_text,
        ]
        context_parts.append("\n".join(part))

    return "\n\n".join(context_parts)


def _format_history_for_prompt(history: list[dict[str, Any]], limit: int = CHAT_HISTORY_LIMIT) -> str:
    if not history:
        return ""

    selected = history[-limit:]
    lines: list[str] = []

    for msg in selected:
        role = msg.get("role", "user")
        message = normalize_whitespace(msg.get("message", ""))
        if not message:
            continue

        role_label = "用户" if role == "user" else "助手" if role == "assistant" else "系统"
        lines.append(f"{role_label}：{message}")

    return "\n".join(lines).strip()


def rewrite_query_with_history(history: list[dict[str, Any]], question: str) -> str:
    question = normalize_whitespace(question)
    if not question:
        return ""

    if not history:
        return question

    history_text = _format_history_for_prompt(history, limit=4)
    if not history_text:
        return question

    system_prompt = (
        "你是一个查询改写助手。"
        "你的任务是把用户当前问题改写成适合知识库检索的独立查询。"
        "要求："
        "1. 保留原意，不要扩展无关信息；"
        "2. 若当前问题依赖上文代词或省略（如“它”“这个”“第二点”），要补全成完整表达；"
        "3. 只输出改写后的单句查询，不要解释。"
    )
    user_prompt = (
        f"对话历史：\n{history_text}\n\n"
        f"当前问题：\n{question}\n\n"
        f"请输出改写后的检索查询："
    )

    try:
        rewritten = chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)
        rewritten = normalize_whitespace(rewritten)
        return rewritten or question
    except Exception:
        return question


def _build_answer_prompt(
    question: str,
    context: str,
    history_text: str = "",
) -> tuple[str, str]:
    system_prompt = (
        "你是一个基于知识库回答问题的助手。"
        "你必须优先依据提供的上下文回答。"
        "如果上下文不足以支持结论，要明确说“知识库中没有足够信息支持该结论”。"
        "不要编造页码、来源或事实。"
        "回答要准确、简洁、中文输出。"
    )

    user_parts = []
    if history_text:
        user_parts.append("对话历史：")
        user_parts.append(history_text)
        user_parts.append("")

    user_parts.append("知识库上下文：")
    user_parts.append(context or "（无可用上下文）")
    user_parts.append("")
    user_parts.append(f"问题：{question}")
    user_parts.append("")
    user_parts.append("请直接给出答案。")

    user_prompt = "\n".join(user_parts)
    return system_prompt, user_prompt


def _build_structured_answer_prompt(
    question: str,
    context: str,
    history_text: str = "",
) -> tuple[str, str]:
    system_prompt = (
        "你是一个基于知识库回答问题的助手。"
        "你必须严格依据上下文作答，不要编造。"
        "输出必须是合法 JSON，且只输出 JSON，不要有额外解释。"
        'JSON 格式如下：{"answer":"","summary":"","key_points":[],"confidence":0.0}'
        "其中 confidence 为 0 到 1 之间的小数。"
        "如果上下文不足，answer 和 summary 中要明确说明信息不足。"
    )

    user_parts = []
    if history_text:
        user_parts.append("对话历史：")
        user_parts.append(history_text)
        user_parts.append("")

    user_parts.append("知识库上下文：")
    user_parts.append(context or "（无可用上下文）")
    user_parts.append("")
    user_parts.append(f"问题：{question}")
    user_parts.append("")
    user_parts.append("请输出 JSON：")

    user_prompt = "\n".join(user_parts)
    return system_prompt, user_prompt


def _safe_parse_structured_answer(raw_text: str) -> dict[str, Any]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {
            "answer": "",
            "summary": "",
            "key_points": [],
            "confidence": 0.0,
        }

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, re.S)
    if fenced:
        raw_text = fenced.group(1).strip()

    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return {
                "answer": normalize_whitespace(data.get("answer", "")),
                "summary": normalize_whitespace(data.get("summary", "")),
                "key_points": data.get("key_points") if isinstance(data.get("key_points"), list) else [],
                "confidence": float(data.get("confidence", 0.0) or 0.0),
            }
    except Exception:
        pass

    return {
        "answer": normalize_whitespace(raw_text),
        "summary": "",
        "key_points": [],
        "confidence": 0.0,
    }


def _estimate_confidence(retrieved_chunks: list[dict[str, Any]], answer_text: str) -> float:
    if not retrieved_chunks:
        return 0.05

    top_score = max([to_float(c.get("score")) for c in retrieved_chunks] + [0.0])
    avg_top3 = sum(to_float(c.get("score")) for c in retrieved_chunks[:3]) / max(
        1, min(3, len(retrieved_chunks))
    )

    confidence = 0.55 * top_score + 0.45 * avg_top3
    if "没有足够信息" in (answer_text or ""):
        confidence *= 0.65

    return round(max(0.0, min(1.0, confidence)), 4)


def _build_sources(
    retrieved_chunks: list[dict[str, Any]],
    highlight: bool,
    highlight_terms: list[str],
    limit: int = 5,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []

    for chunk in retrieved_chunks[:limit]:
        source = _normalize_chunk_source(chunk)
        quote = normalize_whitespace(source.get("quote", ""))
        source["quote"] = _truncate_text(quote, max_len=260)

        if highlight and QA_HIGHLIGHT_ENABLE:
            source["highlight_spans"] = _build_highlight_spans(quote, highlight_terms)

        sources.append(source)

    return sources


def _maybe_update_session_summary(session_id: str) -> None:
    session = get_chat_session(session_id)
    if not session:
        return

    history = get_chat_messages(session_id, limit=CHAT_SUMMARY_TRIGGER_TURNS)
    if len(history) < CHAT_SUMMARY_TRIGGER_TURNS:
        return

    summary_exists = normalize_whitespace(session.get("summary_text", "") or "")
    if summary_exists:
        return

    history_text = _format_history_for_prompt(history, limit=CHAT_SUMMARY_TRIGGER_TURNS)
    if not history_text:
        return

    try:
        summary = summarize_text(history_text)
        summary = normalize_whitespace(summary)
        if summary:
            update_chat_session(session_id=session_id, summary_text=summary)
    except Exception:
        pass


def answer_question(
    question: str,
    top_k: int = 5,
    response_mode: str = "text",
    highlight: bool = True,
    session_id: str | None = None,
    use_chat_context: bool = True,
) -> dict[str, Any]:
    question = normalize_whitespace(question)
    if not question:
        return {
            "question": "",
            "rewritten_query": "",
            "answer": "问题不能为空。",
            "structured": None,
            "sources": [],
            "retrieved_chunks": [],
            "confidence": 0.0,
            "session_id": session_id,
        }

    if not session_id:
        session_id = _new_session_id()
        create_chat_session(session_id=session_id)

    history = get_chat_messages(session_id, limit=CHAT_HISTORY_LIMIT) if use_chat_context else []
    rewritten_query = rewrite_query_with_history(history, question) if use_chat_context else question

    raw_retrieved_chunks = retrieve_chunks(rewritten_query, top_k=top_k)
    retrieved_chunks = [_normalize_retrieved_chunk(chunk) for chunk in raw_retrieved_chunks]

    context = assemble_context(raw_retrieved_chunks, max_chunks=QA_MAX_CONTEXT_CHUNKS)
    history_text = _format_history_for_prompt(history, limit=CHAT_HISTORY_LIMIT) if use_chat_context else ""

    structured = None
    answer_text = ""

    if response_mode == "structured" and QA_STRUCTURED_ENABLE:
        system_prompt, user_prompt = _build_structured_answer_prompt(
            question=question,
            context=context,
            history_text=history_text,
        )
        raw_output = chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)
        parsed = _safe_parse_structured_answer(raw_output)

        answer_text = parsed.get("answer", "") or ""
        structured = {
            "answer": answer_text,
            "summary": parsed.get("summary", "") or "",
            "key_points": parsed.get("key_points", []) or [],
            "sources": [],
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        }
    else:
        system_prompt, user_prompt = _build_answer_prompt(
            question=question,
            context=context,
            history_text=history_text,
        )
        answer_text = normalize_whitespace(
            chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)
        )

    highlight_terms = _extract_query_terms(question, rewritten_query, raw_retrieved_chunks)
    sources = _build_sources(
        retrieved_chunks=raw_retrieved_chunks,
        highlight=highlight,
        highlight_terms=highlight_terms,
        limit=min(5, top_k),
    )

    confidence = _estimate_confidence(raw_retrieved_chunks, answer_text)

    if structured is not None:
        structured["sources"] = sources
        if not structured.get("confidence"):
            structured["confidence"] = confidence

    insert_chat_message(
        session_id=session_id,
        role="user",
        message=question,
        rewritten_query=rewritten_query,
        metadata_json={"top_k": top_k, "response_mode": response_mode},
    )
    insert_chat_message(
        session_id=session_id,
        role="assistant",
        message=answer_text,
        rewritten_query=rewritten_query,
        sources_json=sources,
        metadata_json={"confidence": confidence, "response_mode": response_mode},
    )

    session = get_chat_session(session_id)
    if session and not session.get("title"):
        update_chat_session(session_id=session_id, title=_truncate_text(question, 80))

    _maybe_update_session_summary(session_id)

    return {
        "question": question,
        "rewritten_query": rewritten_query,
        "answer": answer_text,
        "structured": structured,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "confidence": confidence,
        "session_id": session_id,
    }


def summarize_document(document_id: int) -> dict[str, Any]:
    row = get_document_by_id(document_id)
    if not row:
        raise ValueError("document not found")

    title = row.get("title", "") or ""
    content = normalize_whitespace(row.get("content") or row.get("raw_text") or "")

    if not content:
        summary = "文档内容为空，无法摘要。"
    else:
        summary = summarize_text(content)

    return {
        "document_id": document_id,
        "title": title,
        "summary": normalize_whitespace(summary),
    }


def get_chat_history(session_id: str, limit: int = 20) -> dict[str, Any]:
    session = get_chat_session(session_id)
    messages = get_chat_messages(session_id, limit=limit)

    normalized_messages: list[dict[str, Any]] = []
    for msg in messages:
        normalized_messages.append(
            {
                "id": msg.get("id"),
                "session_id": msg.get("session_id"),
                "role": msg.get("role"),
                "message": msg.get("message"),
                "rewritten_query": msg.get("rewritten_query"),
                "sources": msg.get("sources_json") or [],
                "metadata": msg.get("metadata_json") or {},
                "created_at": msg.get("created_at"),
            }
        )

    return {
        "session": {
            "session_id": safe_get(session, "session_id"),
            "title": safe_get(session, "title"),
            "summary_text": safe_get(session, "summary_text"),
            "metadata": safe_get(session, "metadata_json", {}) or {},
            "created_at": safe_get(session, "created_at"),
            "updated_at": safe_get(session, "updated_at"),
        }
        if session
        else None,
        "messages": normalized_messages,
    }