from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from app.db import get_chat_messages, get_chat_session
from app.services.common import normalize_whitespace, safe_get, to_float
from app.services.llm_service import chat_completion, summarize_text
from app.services.qa_service import (
    assemble_context as _assemble_context,
    rewrite_query_with_history as _rewrite_query_with_history,
)
from app.services.retrieval_service import retrieve_chunks
from app.tools.base import ToolExecutionError, require_field, run_tool
from app.tools.schemas import (
    AnswerSource,
    KBAnswerQuestionInput,
    KBAnswerQuestionOutput,
    KBAssembleContextInput,
    KBAssembleContextOutput,
    KBRewriteQueryInput,
    KBRewriteQueryOutput,
    KBGenerateAnswerInput,
    KBGenerateAnswerOutput,
    SourceHighlightSpan,
)


TOOL_REWRITE_QUERY = "kb_rewrite_query"
TOOL_ASSEMBLE_CONTEXT = "kb_assemble_context"
TOOL_GENERATE_ANSWER = "kb_generate_answer"
TOOL_ANSWER_QUESTION = "kb_answer_question"

CHAT_HISTORY_LIMIT = 6
QA_MAX_CONTEXT_CHUNKS = 6
QA_STRUCTURED_ENABLE = True
QA_HIGHLIGHT_ENABLE = True


def _page_label(page_start: Any, page_end: Any) -> str:
    if page_start is None and page_end is None:
        return "-"
    if page_start is not None and page_end is not None:
        if page_start == page_end:
            return str(page_start)
        return f"{page_start}-{page_end}"
    return str(page_start if page_start is not None else page_end)


def _truncate_text(text: str, max_len: int = 220) -> str:
    text = normalize_whitespace(text or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _normalize_chunk_source(chunk: Dict[str, Any]) -> AnswerSource:
    return AnswerSource(
        chunk_id=chunk.get("chunk_id"),
        document_id=chunk.get("document_id"),
        title=chunk.get("title", "") or "",
        section_title=chunk.get("section_title", "") or "",
        section_path=chunk.get("section_path", "") or "",
        page_start=chunk.get("page_start"),
        page_end=chunk.get("page_end"),
        quote=normalize_whitespace(chunk.get("chunk_text", "") or chunk.get("search_text", "") or ""),
        score=to_float(chunk.get("score")),
        highlight_spans=[],
    )


def _build_highlight_spans(text: str, terms: List[str]) -> List[Dict[str, Any]]:
    text = text or ""
    if not text or not terms:
        return []

    spans: List[tuple[int, int]] = []
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
    merged: List[List[int]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return [
        {"start": start, "end": end, "text": text[start:end]}
        for start, end in merged
    ]


def _format_history_for_prompt(history: List[Dict[str, Any]], limit: int = CHAT_HISTORY_LIMIT) -> str:
    if not history:
        return ""
    selected = history[-limit:]
    lines: List[str] = []
    for msg in selected:
        role = msg.get("role", "user")
        message = normalize_whitespace(msg.get("message") or msg.get("content") or "")
        if not message:
            continue
        role_label = "用户" if role == "user" else "助手" if role == "assistant" else "系统"
        lines.append(f"{role_label}：{message}")
    return "\n".join(lines).strip()


def _build_answer_prompt(question: str, context: str, history_text: str = "") -> tuple[str, str]:
    system_prompt = (
        "你是一个基于知识库回答问题的助手。"
        "你必须优先依据提供的上下文回答。"
        "如果上下文不足以支持结论，要明确说"知识库中没有足够信息支持该结论"。"
        "不要编造页码、来源或事实。"
        "回答要准确、简洁、中文输出。"
    )
    user_parts = []
    if history_text:
        user_parts.extend(["对话历史：", history_text, ""])
    user_parts.extend(["知识库上下文：", context or "（无可用上下文）", "", f"问题：{question}", "", "请直接给出答案。"])
    return system_prompt, "\n".join(user_parts)


def _build_structured_answer_prompt(question: str, context: str, history_text: str = "") -> tuple[str, str]:
    system_prompt = (
        "你是一个基于知识库回答问题的助手。"
        "你必须严格依据上下文作答，不要编造。"
        "输出必须是合法 JSON，且只输出 JSON，不要有额外解释。"
        '{"answer":"","summary":"","key_points":[],"confidence":0.0}'
        "其中 confidence 为 0 到 1 之间的小数。"
        "如果上下文不足，answer 和 summary 中要明确说明信息不足。"
    )
    user_parts = []
    if history_text:
        user_parts.extend(["对话历史：", history_text, ""])
    user_parts.extend(["知识库上下文：", context or "（无可用上下文）", "", f"问题：{question}", "", "请输出 JSON："])
    return system_prompt, "\n".join(user_parts)


def _safe_parse_structured_answer(raw_text: str) -> Dict[str, Any]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {"answer": "", "summary": "", "key_points": [], "confidence": 0.0}
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
    return {"answer": normalize_whitespace(raw_text), "summary": "", "key_points": [], "confidence": 0.0}


def _estimate_confidence(retrieved_chunks: List[Dict[str, Any]], answer_text: str) -> float:
    if not retrieved_chunks:
        return 0.05
    top_score = max([to_float(c.get("score")) for c in retrieved_chunks] + [0.0])
    avg_top3 = sum(to_float(c.get("score")) for c in retrieved_chunks[:3]) / max(1, min(3, len(retrieved_chunks)))
    confidence = 0.55 * top_score + 0.45 * avg_top3
    if "没有足够信息" in (answer_text or ""):
        confidence *= 0.65
    return round(max(0.0, min(1.0, confidence)), 4)


def _extract_query_terms(question: str, rewritten_query: str | None, retrieved_chunks: List[Dict[str, Any]]) -> List[str]:
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


def _build_sources(retrieved_chunks: List[Dict[str, Any]], highlight: bool, highlight_terms: List[str], limit: int = 5) -> List[AnswerSource]:
    sources: List[AnswerSource] = []
    for chunk in retrieved_chunks[:limit]:
        source = _normalize_chunk_source(chunk)
        quote = normalize_whitespace(source.quote or "")
        source.quote = _truncate_text(quote, max_len=260)
        if highlight and QA_HIGHLIGHT_ENABLE:
            source.highlight_spans = [
                SourceHighlightSpan(**span) for span in _build_highlight_spans(quote, highlight_terms)
            ]
        sources.append(source)
    return sources


# -------------------------
# Tool implementations
# -------------------------


def kb_rewrite_query(input_data: KBAnswerQuestionInput | Dict[str, Any]) -> Dict[str, Any]:
    payload = input_data if isinstance(input_data, KBRewriteQueryInput) else KBRewriteQueryInput(**input_data)

    def _execute() -> Dict[str, Any]:
        question = normalize_whitespace(payload.question)
        if not question:
            raise ToolExecutionError("EMPTY_QUESTION", "question cannot be empty")

        history: List[Dict[str, Any]] = []
        if payload.use_history and payload.session_id:
            rows = get_chat_messages(payload.session_id, limit=CHAT_HISTORY_LIMIT) or []
            history = [
                {"role": r.get("role", ""), "message": r.get("message") or r.get("content", "")}
                for r in rows
                if r.get("role") in {"user", "assistant"}
            ]

        rewritten_query = _rewrite_query_with_history(history, question) if history else question

        return KBRewriteQueryOutput(
            original_question=question,
            rewritten_query=rewritten_query,
            used_history=payload.use_history and bool(history),
        ).model_dump()

    return run_tool(TOOL_REWRITE_QUERY, _execute)


def kb_assemble_context(input_data: KBAssembleContextInput | Dict[str, Any]) -> Dict[str, Any]:
    payload = input_data if isinstance(input_data, KBAssembleContextInput) else KBAssembleContextInput(**input_data)

    def _execute() -> Dict[str, Any]:
        # Convert SearchHit back to raw chunk dict for assemble_context
        raw_chunks: List[Dict[str, Any]] = []
        for hit in payload.hits:
            chunk: Dict[str, Any] = {
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "title": hit.title,
                "chunk_index": hit.chunk_index,
                "chunk_type": hit.chunk_type,
                "section_path": hit.section_path,
                "section_title": hit.section_title,
                "page_start": hit.page_start,
                "page_end": hit.page_end,
                "chunk_text": hit.text,
                "search_text": hit.text,
                "score": hit.score,
            }
            raw_chunks.append(chunk)

        context = _assemble_context(raw_chunks, max_chunks=payload.max_chunks)
        sources = _build_sources(raw_chunks, highlight=False, highlight_terms=[], limit=payload.max_chunks)

        return KBAssembleContextOutput(
            context=context,
            chunk_count=len(raw_chunks),
            sources=sources,
        ).model_dump()

    return run_tool(TOOL_ASSEMBLE_CONTEXT, _execute)


def kb_generate_answer(input_data: KBGenerateAnswerInput | Dict[str, Any]) -> Dict[str, Any]:
    payload = input_data if isinstance(input_data, KBGenerateAnswerInput) else KBGenerateAnswerInput(**input_data)

    def _execute() -> Dict[str, Any]:
        question = normalize_whitespace(payload.question)
        if not question:
            raise ToolExecutionError("EMPTY_QUESTION", "question cannot be empty")

        context = payload.context or "（无可用上下文）"
        history_text = payload.history_text or ""

        answer_text = ""
        confidence = None
        key_points: List[str] = []
        summary = ""

        if payload.response_mode == "structured" and QA_STRUCTURED_ENABLE:
            system_prompt, user_prompt = _build_structured_answer_prompt(question, context, history_text)
            raw_output = chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)
            parsed = _safe_parse_structured_answer(raw_output)
            answer_text = parsed.get("answer", "") or ""
            confidence = parsed.get("confidence", 0.0)
            key_points = parsed.get("key_points", []) or []
            summary = parsed.get("summary", "") or ""
        else:
            system_prompt, user_prompt = _build_answer_prompt(question, context, history_text)
            answer_text = normalize_whitespace(chat_completion(system_prompt=system_prompt, user_prompt=user_prompt))

        return KBGenerateAnswerOutput(
            answer=answer_text,
            confidence=confidence,
            key_points=key_points,
            summary=summary,
            sources=[],
        ).model_dump()

    return run_tool(TOOL_GENERATE_ANSWER, _execute)


def kb_answer_question(input_data: KBAnswerQuestionInput | Dict[str, Any]) -> Dict[str, Any]:
    payload = input_data if isinstance(input_data, KBAnswerQuestionInput) else KBAnswerQuestionInput(**input_data)

    def _execute() -> Dict[str, Any]:
        question = normalize_whitespace(payload.question)
        if not question:
            raise ToolExecutionError("EMPTY_QUESTION", "question cannot be empty")

        # Get history
        history: List[Dict[str, Any]] = []
        if payload.use_chat_context and payload.session_id:
            rows = get_chat_messages(payload.session_id, limit=CHAT_HISTORY_LIMIT) or []
            history = [
                {"role": r.get("role", ""), "message": r.get("message") or r.get("content", "")}
                for r in rows
                if r.get("role") in {"user", "assistant"}
            ]

        # Rewrite query
        rewritten_query = _rewrite_query_with_history(history, question) if history else question

        # Retrieve chunks
        raw_chunks = retrieve_chunks(rewritten_query, top_k=payload.top_k) or []

        # Assemble context
        context = _assemble_context(raw_chunks, max_chunks=QA_MAX_CONTEXT_CHUNKS)
        history_text = _format_history_for_prompt(history) if payload.use_chat_context else ""

        # Generate answer
        answer_text = ""
        structured: Dict[str, Any] | None = None
        key_points: List[str] = []
        summary = ""

        if payload.response_mode == "structured" and QA_STRUCTURED_ENABLE:
            system_prompt, user_prompt = _build_structured_answer_prompt(question, context, history_text)
            raw_output = chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)
            parsed = _safe_parse_structured_answer(raw_output)
            answer_text = parsed.get("answer", "") or ""
            key_points = parsed.get("key_points", []) or []
            summary = parsed.get("summary", "") or ""
            confidence = parsed.get("confidence", 0.0)
            structured = {
                "answer": answer_text,
                "summary": summary,
                "key_points": key_points,
                "confidence": confidence,
                "sources": [],
            }
        else:
            system_prompt, user_prompt = _build_answer_prompt(question, context, history_text)
            answer_text = normalize_whitespace(chat_completion(system_prompt=system_prompt, user_prompt=user_prompt))
            confidence = _estimate_confidence(raw_chunks, answer_text)

        # Build sources
        highlight_terms = _extract_query_terms(question, rewritten_query, raw_chunks)
        sources = _build_sources(raw_chunks, highlight=payload.highlight, highlight_terms=highlight_terms, limit=min(5, payload.top_k))

        if structured is not None:
            structured["sources"] = [s.model_dump() for s in sources]

        # Convert raw_chunks to SearchHit-like dicts
        retrieved_chunks = [
            {
                "chunk_id": c.get("chunk_id"),
                "document_id": c.get("document_id"),
                "title": c.get("title", "") or "",
                "chunk_index": c.get("chunk_index"),
                "chunk_type": c.get("chunk_type"),
                "section_path": c.get("section_path", "") or "",
                "section_title": c.get("section_title", "") or "",
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "score": c.get("score"),
                "text": c.get("chunk_text", "") or c.get("search_text", "") or "",
            }
            for c in raw_chunks
        ]

        return KBAnswerQuestionOutput(
            question=question,
            rewritten_query=rewritten_query,
            answer=answer_text,
            confidence=confidence,
            structured=structured,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
            session_id=payload.session_id or "",
        ).model_dump()

    return run_tool(TOOL_ANSWER_QUESTION, _execute)
