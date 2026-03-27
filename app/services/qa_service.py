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
from app.services.retrieval_service import retrieve_chunks, enhance_financial_query, classify_query_intent


def _cfg(name: str, default: Any):
    return getattr(config, name, default)


CHAT_HISTORY_LIMIT = int(_cfg("CHAT_HISTORY_LIMIT", 6))
CHAT_SUMMARY_TRIGGER_TURNS = int(_cfg("CHAT_SUMMARY_TRIGGER_TURNS", 12))
QA_MAX_CONTEXT_CHUNKS = int(_cfg("QA_MAX_CONTEXT_CHUNKS", 6))
QA_STRUCTURED_ENABLE = bool(_cfg("QA_STRUCTURED_ENABLE", True))
QA_HIGHLIGHT_ENABLE = bool(_cfg("QA_HIGHLIGHT_ENABLE", True))

# V4 pipeline params — read from app.config directly
V4_ENABLE_ANSWER_VERIFIER = bool(_cfg("V4_ENABLE_ANSWER_VERIFIER", True))
V4_ENABLE_SELF_REFINE = bool(_cfg("V4_ENABLE_SELF_REFINE", False))
V4_MAX_REFINE_ROUNDS = int(_cfg("V4_MAX_REFINE_ROUNDS", 1))
V4_ANSWER_USE_STRUCTURED_OUTPUT = bool(_cfg("V4_ANSWER_USE_STRUCTURED_OUTPUT", True))
V4_NUMERIC_FIRST_FOR_NUMERIC_QUERIES = bool(_cfg("V4_NUMERIC_FIRST_FOR_NUMERIC_QUERIES", True))


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
        "title": chunk.get("title", "") or "",
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
        "rerank_score": to_float(chunk.get("rerank_score")),
        "embedding_score": to_float(chunk.get("embedding_score")),
        "keyword_score": to_float(chunk.get("keyword_score")),
        "bm25_score": to_float(chunk.get("bm25_score")),
        "title_match_score": to_float(chunk.get("title_match_score")),
        "section_match_score": to_float(chunk.get("section_match_score")),
        "coverage_score": to_float(chunk.get("coverage_score")),
        "matched_term_count": chunk.get("matched_term_count"),
        "title": chunk.get("title", "") or "",
        "section_title": chunk.get("section_title", "") or "",
        "section_path": chunk.get("section_path", "") or "",
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "chunk_type": chunk.get("chunk_type"),
        "chunk_text": chunk.get("chunk_text", "") or "",
        "term_hits": chunk.get("term_hits") or {},
        "term_hit_detail": chunk.get("term_hit_detail") or {},
        "is_neighbor": bool(chunk.get("is_neighbor", False)),
        "_retrieval_query": chunk.get("_retrieval_query"),
    }


def assemble_context(chunks: list[dict[str, Any]], max_chunks: int = QA_MAX_CONTEXT_CHUNKS) -> str:
    if not chunks:
        return ""

    context_parts: list[str] = []

    for idx, chunk in enumerate(chunks[:max_chunks], start=1):
        title = chunk.get("title", "") or ""
        section_title = chunk.get("section_title", "") or ""
        section_path = chunk.get("section_path", "") or ""
        page_label = _page_label(chunk.get("page_start"), chunk.get("page_end"))
        chunk_text = normalize_whitespace(chunk.get("search_text") or chunk.get("chunk_text") or "")

        part = [
            f"[Source {idx}]",
            f"Document: {title or '-'}",
            f"Section: {section_title or section_path or '-'}",
            f"Page: {page_label}",
            "Content:",
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

        role_label = "User" if role == "user" else "Assistant" if role == "assistant" else "System"
        lines.append(f"{role_label}: {message}")

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
        "You are a query rewriting assistant. Rewrite the user's current question into "
        "a standalone query suitable for knowledge base retrieval.\n"
        "Requirements:\n"
        "1. Preserve original intent; do not expand with unrelated information.\n"
        "2. If the current question depends on pronouns or ellipsis from context "
        "(e.g. 'it', 'this', 'the second point'), complete the expression.\n"
        "3. Output only the rewritten single-sentence query, no explanation."
    )
    user_prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Current question:\n{question}\n\n"
        f"Rewritten retrieval query:"
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
        "You are a financial document Q&A assistant drawing from a trusted knowledge base.\n\n"
        "PRIORITIES (in order):\n"
        "1. NUMBERS FIRST: Always lead with specific figures -- percentages, dollar amounts, "
        "ratios, units, years. Cite them verbatim from context.\n"
        "2. ENGLISH OUTPUT: Answer in English unless the user explicitly asks in Chinese.\n"
        "3. DRIVER/IMPACT QUESTIONS: If the question asks 'what drove', 'what caused', "
        "'what is the impact of', structure your answer as:\n"
        "   - Positive drivers (+X% / $X / X basis points)\n"
        "   - Negative drivers (-X% / -$X / -X basis points)\n"
        "   - Quantified impact (X% of total change attributable to this factor)\n"
        "4. NO VAGUE REFUSALS: If the context contains relevant evidence, give a grounded answer. "
        "Only say 'insufficient information' if the context genuinely does not address the question. "
        "Do NOT use templates like 'no information in knowledge base' when evidence exists.\n"
        "5. ACCURACY: Never fabricate page numbers, sources, or facts. "
        "If context does not support a conclusion, state what IS supported.\n"
        "6. BREVITY: Short and direct. One short paragraph unless complexity demands more."
    )

    user_parts = []
    if history_text:
        user_parts.append("Conversation history:")
        user_parts.append(history_text)
        user_parts.append("")

    user_parts.append("Knowledge base context:")
    user_parts.append(context or "(no context available)")
    user_parts.append("")
    user_parts.append(f"Question: {question}")
    user_parts.append("")
    user_parts.append("Answer:")

    user_prompt = "\n".join(user_parts)
    return system_prompt, user_prompt


def _build_structured_answer_prompt(
    question: str,
    context: str,
    history_text: str = "",
) -> tuple[str, str]:
    system_prompt = (
        "You are a financial document Q&A assistant drawing from a trusted knowledge base.\n\n"
        "PRIORITIES:\n"
        "1. NUMBERS FIRST: Lead with specific figures -- percentages, dollar amounts, ratios.\n"
        "2. ENGLISH OUTPUT: Answer in English unless the user explicitly asks in Chinese.\n"
        "3. DRIVER/IMPACT QUESTIONS: For 'what drove' / 'what caused' questions, use these JSON fields:\n"
        "   {'answer':'<brief direct answer with numbers first>',\n"
        "    'positive_drivers':['<driver 1 with quantified impact>',...],\n"
        "    'negative_drivers':['<driver 1 with quantified impact>',...],\n"
        "    'key_points':['<supporting fact 1>',...],\n"
        "    'confidence':0.0}\n"
        "4. NO VAGUE REFUSALS: Only state insufficient information when context genuinely lacks evidence.\n"
        "5. Output ONLY valid JSON. No markdown, no explanation."
    )

    user_parts = []
    if history_text:
        user_parts.append("Conversation history:")
        user_parts.append(history_text)
        user_parts.append("")

    user_parts.append("Knowledge base context:")
    user_parts.append(context or "(no context available)")
    user_parts.append("")
    user_parts.append(f"Question: {question}")
    user_parts.append("")
    user_parts.append("Output JSON:")

    user_prompt = "\n".join(user_parts)
    return system_prompt, user_prompt


def _safe_parse_structured_answer(raw_text: str) -> dict[str, Any]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {
            "answer": "",
            "positive_drivers": [],
            "negative_drivers": [],
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
                "positive_drivers": data.get("positive_drivers") if isinstance(data.get("positive_drivers"), list) else [],
                "negative_drivers": data.get("negative_drivers") if isinstance(data.get("negative_drivers"), list) else [],
                "key_points": data.get("key_points") if isinstance(data.get("key_points"), list) else [],
                "confidence": float(data.get("confidence", 0.0) or 0.0),
            }
    except Exception:
        pass

    return {
        "answer": normalize_whitespace(raw_text),
        "positive_drivers": [],
        "negative_drivers": [],
        "key_points": [],
        "confidence": 0.0,
    }


def _extract_cited_evidence_ids(answer_text: str, retrieved_chunks: list[dict[str, Any]]) -> list[str]:
    """
    Extract cited evidence IDs from answer text.
    Matches patterns like [E1], [E2], source 1, source 2, etc.
    """
    if not answer_text or not retrieved_chunks:
        return []

    cited: list[str] = []
    answer_lower = answer_text.lower()

    for i, chunk in enumerate(retrieved_chunks[:5]):
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or f"E{i+1}")
        # Check if chunk_id or E{i+1} is mentioned in answer
        if chunk_id.lower() in answer_lower or f"e{i+1}" in answer_lower or f"source {i+1}" in answer_lower:
            if chunk_id not in cited:
                cited.append(chunk_id)

    return cited


def _augment_answer_with_citations(answer_text: str, cited_ids: list[str]) -> str:
    """Add evidence citation suffix to answer text if not already present."""
    if not answer_text or not cited_ids:
        return answer_text

    # Check if answer already has citations
    import re
    if re.search(r"\[E\d+\]|\[Source \d+\]|\[\d+\]", answer_text):
        return answer_text

    citation_str = " [Evidence: " + ", ".join(f"[{cid}]" for cid in cited_ids[:3]) + "]"
    return answer_text + citation_str


def _estimate_confidence(retrieved_chunks: list[dict[str, Any]], answer_text: str) -> float:
    if not retrieved_chunks:
        return 0.05

    top_score = max([to_float(c.get("score")) for c in retrieved_chunks] + [0.0])
    avg_top3 = sum(to_float(c.get("score")) for c in retrieved_chunks[:3]) / max(
        1, min(3, len(retrieved_chunks))
    )

    confidence = 0.55 * top_score + 0.45 * avg_top3
    # Penalize vague refusal language
    refusal_phrases = {"insufficient information", "not enough information", "cannot confirm",
                       "no relevant", "not sufficient", "cannot determine", "insufficient context"}
    answer_lower = (answer_text or "").lower()
    if any(rp in answer_lower for rp in refusal_phrases):
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
    retrieve_top_k: int | None = None,
    enable_query_enhance: bool = False,
    use_multistage: bool = False,
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


    session = get_chat_session(session_id)
    if not session:
        create_chat_session(
            session_id=session_id,
            title=_truncate_text(question, 80),
            metadata={},
    )

    history = get_chat_messages(session_id, limit=CHAT_HISTORY_LIMIT) if use_chat_context else []
    rewritten_query = rewrite_query_with_history(history, question) if use_chat_context else question

    # Two-stage retrieval: retrieve_top_k > top_k means fetch more, rerank, then truncate
    effective_retrieve_top_k = retrieve_top_k if retrieve_top_k is not None else top_k

    # Query enhancement: augment query for retrieval only (answer uses original question)
    enhanced_query = None
    if enable_query_enhance:
        enhanced_query = enhance_financial_query(rewritten_query)

    raw_retrieved_chunks = retrieve_chunks(
        rewritten_query,
        top_k=effective_retrieve_top_k,
        enhanced_query=enhanced_query,
        use_multistage=use_multistage,
    )
    retrieved_chunks = [_normalize_retrieved_chunk(chunk) for chunk in raw_retrieved_chunks]

    context = assemble_context(raw_retrieved_chunks, max_chunks=QA_MAX_CONTEXT_CHUNKS)
    history_text = _format_history_for_prompt(history, limit=CHAT_HISTORY_LIMIT) if use_chat_context else ""

    structured = None
    answer_text = ""
    intent = "unknown"
    try:
        intent = classify_query_intent(question)
    except Exception:
        pass

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
            "positive_drivers": parsed.get("positive_drivers", []) or [],
            "negative_drivers": parsed.get("negative_drivers", []) or [],
            "key_points": parsed.get("key_points", []) or [],
            "sources": [],
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "cited_evidence_ids": [],
            "answer_type": intent,
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

    # V4: Add evidence citations to answer text
    cited_evidence_ids: list[str] = []
    if answer_text and raw_retrieved_chunks:
        cited_evidence_ids = _extract_cited_evidence_ids(answer_text, raw_retrieved_chunks)
        if cited_evidence_ids and V4_ANSWER_USE_STRUCTURED_OUTPUT:
            # Augment answer text with evidence IDs if not already present
            answer_text = _augment_answer_with_citations(answer_text, cited_evidence_ids)

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
        structured["cited_evidence_ids"] = cited_evidence_ids
        if not structured.get("confidence"):
            structured["confidence"] = confidence

    # V4: Answer verification + self-refine
    # Initialize with safe defaults
    verification_result: dict[str, Any] = {
        "is_supported": True,
        "support_level": "high",
        "numeric_consistency": None,
        "citation_adequate": True,
        "failure_reasons": [],
        "missing_requirements": [],
        "summary": "verifier_disabled",
        "method": "disabled",
    }
    refine_result: dict[str, Any] = {
        "refined_answer": "",
        "was_refined": False,
        "refinement_round": 0,
        "refinement_applied": False,
        "trigger_reason": "disabled",
        "missing_requirements_addressed": [],
        "method": "disabled",
    }
    final_answer = answer_text
    verifier_fallback = False
    refine_fallback = False

    if V4_ENABLE_ANSWER_VERIFIER and raw_retrieved_chunks:
        try:
            from app.services.verifier_service import verify_answer
            verification_result = verify_answer(
                query=question,
                draft_answer=answer_text,
                evidence_chunks=raw_retrieved_chunks,
                intent=intent,
            )
            final_answer = answer_text

            # Track if verifier fell back
            verifier_fallback = verification_result.get("method") == "fallback"

            # V4: Self-refine using explicit trigger conditions from refine_service
            if V4_ENABLE_SELF_REFINE:
                from app.services.refine_service import refine_answer as _refine_answer
                refine_result = _refine_answer(
                    query=question,
                    draft_answer=answer_text,
                    verification_result=verification_result,
                    evidence_chunks=raw_retrieved_chunks,
                    round_num=1,
                )
                if refine_result.get("refinement_applied") and refine_result.get("refined_answer"):
                    final_answer = refine_result["refined_answer"]
                elif refine_result.get("method") == "failed":
                    refine_fallback = True
        except Exception:
            # Never let verifier failure crash the pipeline
            verification_result = {
                "is_supported": True,
                "support_level": "high",
                "numeric_consistency": None,
                "citation_adequate": True,
                "failure_reasons": ["verifier_exception"],
                "missing_requirements": [],
                "summary": "verifier_exception_fallback",
                "method": "exception",
            }
            verifier_fallback = True

    insert_chat_message(
        session_id=session_id,
        role="user",
        message=question,
        rewritten_query=rewritten_query,
        metadata={"top_k": top_k, "response_mode": response_mode},
    )
    insert_chat_message(
        session_id=session_id,
        role="assistant",
        message=final_answer,
        rewritten_query=rewritten_query,
        sources=sources,
        metadata={
            "confidence": confidence,
            "response_mode": response_mode,
            "verification": verification_result,
            "refine": refine_result,
            "verifier_fallback": verifier_fallback,
            "refine_fallback": refine_fallback,
        },
    )

    session = get_chat_session(session_id)
    if session and not session.get("title"):
        update_chat_session(session_id=session_id, title=_truncate_text(question, 80))

    _maybe_update_session_summary(session_id)

    return {
        "question": question,
        "rewritten_query": rewritten_query,
        "answer": final_answer,
        "draft_answer": answer_text if answer_text != final_answer else None,
        "structured": structured,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "confidence": confidence,
        "session_id": session_id,
        "query_intent": intent,
        "verification_result": verification_result,
        "refine_result": refine_result,
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
                "sources": msg.get("sources") or [],
                "metadata": msg.get("metadata") or {},
                "created_at": msg.get("created_at"),
            }
        )

    return {
        "session": {
            "session_id": safe_get(session, "session_id"),
            "title": safe_get(session, "title"),
            "summary_text": safe_get(session, "summary_text"),
            "metadata": safe_get(session, "metadata", {}) or {},
            "created_at": safe_get(session, "created_at"),
            "updated_at": safe_get(session, "updated_at"),
        }
        if session
        else None,
        "messages": normalized_messages,
    }