"""
Query understanding: normalization, tokenization, intent classification, financial query enhancement.
"""
from __future__ import annotations

import re
from typing import Any

from app.retrieval._common import normalize_whitespace

from .config import (
    CJK_RE,
    CJK_TOKEN_RE,
    ASCII_TOKEN_RE,
    LEXICAL_CLEAN_RE,
    QUERY_INTENT_DESCRIPTIVE,
    QUERY_INTENT_HYBRID,
    QUERY_INTENT_LIST,
    QUERY_INTENT_NUMERIC,
    QUERY_MIN_TERM_LEN,
    QUERY_STOPWORDS,
    _DESCRIPTIVE_PATTER,
    _LIST_FACT_PATTER,
    _NUMERIC_FACT_PATTER,
)


def _contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def _normalize_lexical_text(text: str) -> str:
    text = normalize_whitespace(text or "")
    text = text.lower()
    text = LEXICAL_CLEAN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_text(text: str) -> list[str]:
    """Full-featured tokenization with CJK sub-tokenization. Used by rewrite_query."""
    if not text:
        return []
    text = normalize_whitespace(text)
    tokens: list[str] = []
    for token in ASCII_TOKEN_RE.findall(text):
        token = token.strip()
        if not token:
            continue
        tokens.append(token.lower())
    for chunk in CJK_TOKEN_RE.findall(text):
        chunk = chunk.strip()
        if not chunk:
            continue
        tokens.append(chunk)
        if len(chunk) >= 2:
            for n in (2, 3, 4):
                if len(chunk) >= n:
                    for i in range(len(chunk) - n + 1):
                        tokens.append(chunk[i : i + n])
    return [t for t in tokens if t]


def _tokenize_query(text: str) -> list[str]:
    """Split query into ASCII/alphanumeric tokens and CJK character groups."""
    if not text:
        return []
    text = normalize_whitespace(text)
    tokens: list[str] = []
    for match in ASCII_TOKEN_RE.finditer(text):
        token = match.group()
        if len(token) >= QUERY_MIN_TERM_LEN:
            tokens.append(token.lower())
    for match in CJK_TOKEN_RE.finditer(text):
        tokens.append(match.group())
    return tokens


def _normalize_mixed_language_query(text: str) -> str:
    """Normalize a mixed Chinese/English query for embedding."""
    text = normalize_whitespace(text or "").strip()
    if not text:
        return ""
    # Remove redundant spaces around CJK chars
    text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)
    text = re.sub(r"([\u4e00-\u9fff])\s+([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])\s+([\u4e00-\u9fff])", r"\1 \2", text)
    return text


def _normalize_query(query: str) -> dict[str, Any]:
    """
    Normalize a query and return a dict with important_terms.
    Keeps full backward compatibility with the original signature.
    """
    normalized = _normalize_mixed_language_query(query)
    terms = _tokenize_query(normalized)
    unique_terms = list(dict.fromkeys(terms))

    important_terms: list[str] = []
    for term in unique_terms:
        if (
            _contains_cjk(term)
            or re.search(r"[_./:-]", term)
            or re.fullmatch(r"[a-z][a-z0-9_./:-]*", term)
            or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", term)
            or len(term) >= 4
        ):
            important_terms.append(term)

    if not important_terms:
        important_terms = unique_terms[:]

    return {
        "raw_query": query,
        "normalized_query": normalized,
        "terms": terms,
        "unique_terms": unique_terms,
        "important_terms": important_terms,
        "contains_cjk": _contains_cjk(normalized),
    }


def classify_query_intent(query: str) -> str:
    """
    Classify query into intent types:
      - numeric_fact  : query asks for a specific number/ratio/amount
      - list_fact     : query asks for items/products/services/segments
      - descriptive   : query asks for explanation/discussion
      - hybrid        : combination or unclear
    """
    q = query.lower()
    numeric_score = len(_NUMERIC_FACT_PATTER.findall(q))
    list_score = len(_LIST_FACT_PATTER.findall(q))
    descriptive_score = len(_DESCRIPTIVE_PATTER.findall(q))

    scores = {
        QUERY_INTENT_NUMERIC: numeric_score,
        QUERY_INTENT_LIST: list_score,
        QUERY_INTENT_DESCRIPTIVE: descriptive_score,
    }
    max_score = max(scores.values())

    if max_score == 0:
        return QUERY_INTENT_HYBRID

    winners = [k for k, v in scores.items() if v == max_score]
    if len(winners) >= 2:
        return QUERY_INTENT_HYBRID
    return winners[0]


def _detect_financial_query(query_info: dict[str, Any]) -> bool:
    """Return True if the query appears to be financial in nature."""
    norm = query_info.get("normalized_query", "")
    cjk = query_info.get("contains_cjk", False)

    financial_keywords_cjk = [
        "10-K", "10k", "年报", "季报", "财务报表", "资产负债表",
        "利润表", "现金流量表", "营收", "净利润", "毛利率", "每股收益",
        "股息", "债务", "资产",
    ]
    financial_keywords_en = [
        "10-k", "annual report", "financial statement", "balance sheet",
        "income statement", "cash flow", "revenue", "net income", "eps",
        "shares outstanding", "liabilities", "assets",
    ]

    text_lower = norm.lower()
    for kw in financial_keywords_cjk + financial_keywords_en:
        if kw in text_lower:
            return True

    if cjk:
        return any(cjk_kw in norm for cjk_kw in financial_keywords_cjk)
    return False


def _query_has_financial_terms(query_info: dict[str, Any]) -> bool:
    """Alias for _detect_financial_query for backward compat."""
    return _detect_financial_query(query_info)


def _build_expanded_query(query_info: dict[str, Any]) -> str:
    """
    Build an expanded query by appending financial context terms
    when the query is detected as financial.
    """
    if not _detect_financial_query(query_info):
        return query_info.get("normalized_query", "")

    expansions = ["financial statements", "annual report", "10-K", "fiscal"]
    query = query_info.get("normalized_query", "")
    for exp in expansions:
        if exp.lower() not in query.lower():
            query = f"{query} {exp}"
    return query


def enhance_financial_query(query: str) -> str:
    """
    Public API: enhance a financial query with context terms.
    Returns the enhanced query string.
    """
    qi = _normalize_query(query)
    return _build_expanded_query(qi)


def rewrite_query(query: str, intent: str | None = None) -> dict[str, Any]:
    """
    Rewrite a user query for better retrieval.

    Returns:
        {
            "original_query": str,
            "rewritten_query": str,
            "intent": str,
            "added_terms": list[str],
        }

    Rules:
      1. Short queries (<= 3 tokens) get strong expansion
      2. Numeric/list queries get financial section context
      3. Descriptive queries get causal/explanatory context
      4. Always strip noise, preserve entity names
    """
    original = query.strip()
    query_lower = original.lower()
    tokens = _tokenize_text(original)

    added_terms: list[str] = []

    if intent is None:
        intent = classify_query_intent(original)

    noise_prefixes = [
        "what is", "what are", "what was", "what were",
        "how much", "how many", "how did", "how does",
        "why did", "why does", "why was", "why are",
        "can you", "could you", "please", "tell me",
    ]
    stripped = query_lower
    for prefix in noise_prefixes:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):].strip()
            break

    stripped_tokens = _tokenize_text(stripped)
    is_short = len(stripped_tokens) <= 3

    rewritten_parts = [original]

    _FINANCIAL_SECTION_TERMS = [
        "10-K", "annual report", "financial statement",
        "item 7", "item 8", "management discussion",
        "balance sheet", "income statement", "cash flow",
        "consolidated financial statements",
    ]

    if intent == QUERY_INTENT_NUMERIC:
        if is_short:
            rewritten_parts.extend(_FINANCIAL_SECTION_TERMS[:5])
            added_terms.extend(_FINANCIAL_SECTION_TERMS[:5])
        metric_terms = [
            "ratio", "percentage", "million", "billion",
            "assets", "liabilities", "revenue", "income",
            "cash flow", "tax rate", "margin",
        ]
        for t in metric_terms:
            if t not in query_lower:
                rewritten_parts.append(t)
                added_terms.append(t)

    elif intent == QUERY_INTENT_LIST:
        if is_short:
            list_terms = [
                "products", "services", "segments", "business",
                "overview", "item 1", "our business",
            ]
            for t in list_terms:
                if t not in query_lower:
                    rewritten_parts.append(t)
                    added_terms.append(t)

    elif intent == QUERY_INTENT_DESCRIPTIVE:
        if is_short:
            desc_terms = [
                "management discussion", "analysis", "item 7",
                "overview", "results of operations",
            ]
            for t in desc_terms:
                if t not in query_lower:
                    rewritten_parts.append(t)
                    added_terms.append(t)

    if is_short and len(stripped_tokens) <= 2:
        for t in _FINANCIAL_SECTION_TERMS:
            if t not in query_lower and len(added_terms) < 4:
                rewritten_parts.append(t)
                added_terms.append(t)

    rewritten = " ".join(rewritten_parts)

    return {
        "original_query": original,
        "rewritten_query": rewritten,
        "intent": intent,
        "added_terms": added_terms,
    }
