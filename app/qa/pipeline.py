"""
Main QA pipeline: answer_question entry point with V4 verifier + refine.
"""
from __future__ import annotations

from typing import Any

from app.db import (
    create_chat_session,
    get_chat_messages,
    get_chat_session,
    insert_chat_message,
    update_chat_session,
)
from app.services.common import normalize_whitespace
from app.services.llm_service import chat_completion
from app.retrieval.query_understanding import enhance_financial_query, classify_query_intent
from app.retrieval.service import retrieve_chunks

from .config import (
    CHAT_HISTORY_LIMIT,
    QA_MAX_CONTEXT_CHUNKS,
    QA_STRUCTURED_ENABLE,
    V4_ANSWER_USE_STRUCTURED_OUTPUT,
    V4_ENABLE_ANSWER_VERIFIER,
    V4_ENABLE_SELF_REFINE,
    _extract_query_terms,
    _new_session_id,
    _normalize_retrieved_chunk,
    _truncate_text,
)
from .context import _format_history_for_prompt, assemble_context
from .prompts import (
    _augment_answer_with_citations,
    _build_answer_prompt,
    _build_sources,
    _build_structured_answer_prompt,
    _estimate_confidence,
    _extract_cited_evidence_ids,
    _safe_parse_structured_answer,
)
from .session import _maybe_update_session_summary


def rewrite_query_with_history(history: list[dict[str, Any]], question: str) -> str:
    """Rewrite a question using conversation history for context."""
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
    """
    Full RAG pipeline: question → rewrite → retrieve → build context → generate answer.
    """
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

    effective_retrieve_top_k = retrieve_top_k if retrieve_top_k is not None else top_k

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

    cited_evidence_ids: list[str] = []
    if answer_text and raw_retrieved_chunks:
        cited_evidence_ids = _extract_cited_evidence_ids(answer_text, raw_retrieved_chunks)
        if cited_evidence_ids and V4_ANSWER_USE_STRUCTURED_OUTPUT:
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
            verifier_fallback = verification_result.get("method") == "fallback"

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
