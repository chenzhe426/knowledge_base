import math
import re
from collections import Counter, defaultdict
from typing import Any

import app.config as config
from app.db import (
    get_chunks_by_ids,
    get_neighbor_chunks,
    search_chunks_boolean,
    search_chunks_fulltext,
)
from app.services.common import (
    normalize_embedding,
    normalize_whitespace,
    safe_get,
    safe_json_loads,
    to_float,
)
from app.services.vector_store import vector_store
from app.services.reranker_service import rerank_with_llm


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./:-]*|[0-9]+(?:\.[0-9]+)?")
CJK_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+")
LEXICAL_CLEAN_RE = re.compile(r"[^\w\u4e00-\u9fff]+")


def _cfg(name: str, default: Any):
    return getattr(config, name, default)


DEFAULT_TOP_K = _cfg("DEFAULT_TOP_K", 5)

HYBRID_VECTOR_TOP_K = int(_cfg("HYBRID_VECTOR_TOP_K", 20))
HYBRID_KEYWORD_TOP_K = int(_cfg("HYBRID_KEYWORD_TOP_K", 20))
HYBRID_LEXICAL_FETCH_K = int(_cfg("HYBRID_LEXICAL_FETCH_K", 120))
HYBRID_BOOLEAN_FETCH_K = int(_cfg("HYBRID_BOOLEAN_FETCH_K", 80))
HYBRID_HYDRATE_TOP_K = int(_cfg("HYBRID_HYDRATE_TOP_K", 160))

RETRIEVAL_WEIGHT_EMBEDDING = float(_cfg("RETRIEVAL_WEIGHT_EMBEDDING", 0.45))
RETRIEVAL_WEIGHT_KEYWORD = float(_cfg("RETRIEVAL_WEIGHT_KEYWORD", 0.20))
RETRIEVAL_WEIGHT_TITLE = float(_cfg("RETRIEVAL_WEIGHT_TITLE", 0.15))
RETRIEVAL_WEIGHT_SECTION = float(_cfg("RETRIEVAL_WEIGHT_SECTION", 0.10))
RETRIEVAL_WEIGHT_BM25 = float(_cfg("RETRIEVAL_WEIGHT_BM25", 0.10))

TITLE_MATCH_WEIGHT = float(_cfg("TITLE_MATCH_WEIGHT", 0.8))
SECTION_MATCH_WEIGHT = float(_cfg("SECTION_MATCH_WEIGHT", 0.6))
KEYWORD_EXACT_MATCH_WEIGHT = float(_cfg("KEYWORD_EXACT_MATCH_WEIGHT", 1.0))
KEYWORD_SUBSTRING_MATCH_WEIGHT = float(_cfg("KEYWORD_SUBSTRING_MATCH_WEIGHT", 0.55))

RETRIEVAL_DEDUP_SIM_THRESHOLD = float(_cfg("RETRIEVAL_DEDUP_SIM_THRESHOLD", 0.82))
RETRIEVAL_MAX_SAME_SECTION = int(_cfg("RETRIEVAL_MAX_SAME_SECTION", 2))
RETRIEVAL_ENABLE_NEIGHBOR_EXPANSION = bool(_cfg("RETRIEVAL_ENABLE_NEIGHBOR_EXPANSION", True))
RETRIEVAL_NEIGHBOR_WINDOW = int(_cfg("RETRIEVAL_NEIGHBOR_WINDOW", 1))

# --- New rerank signal weights (P0) ---
RETRIEVAL_NUMERIC_BOOST_WEIGHT = float(_cfg("RETRIEVAL_NUMERIC_BOOST_WEIGHT", 0.08))
RETRIEVAL_TABLE_BOOST_WEIGHT = float(_cfg("RETRIEVAL_TABLE_BOOST_WEIGHT", 0.06))
RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT = float(_cfg("RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT", 0.07))
RETRIEVAL_ANTI_NOISE_PENALTY_WEIGHT = float(_cfg("RETRIEVAL_ANTI_NOISE_PENALTY_WEIGHT", 0.04))
RETRIEVAL_PAGE_CLUSTER_ALPHA = float(_cfg("RETRIEVAL_PAGE_CLUSTER_ALPHA", 0.15))

# --- V3 Multi-stage retrieval config ---
RETRIEVAL_USE_MULTISTAGE = bool(_cfg("RETRIEVAL_USE_MULTISTAGE", False))
RETRIEVAL_CANDIDATE_DOCS = int(_cfg("RETRIEVAL_CANDIDATE_DOCS", 3))
RETRIEVAL_CANDIDATE_SECTIONS = int(_cfg("RETRIEVAL_CANDIDATE_SECTIONS", 5))
RETRIEVAL_CHUNKS_PER_SECTION = int(_cfg("RETRIEVAL_CHUNKS_PER_SECTION", 5))
RETRIEVAL_SECTION_PAGE_WINDOW = int(_cfg("RETRIEVAL_SECTION_PAGE_WINDOW", 3))
RETRIEVAL_SECTION_EMBEDDING_TOP_K = int(_cfg("RETRIEVAL_SECTION_EMBEDDING_TOP_K", 10))
RETRIEVAL_HYBRID_W_DENSE = float(_cfg("RETRIEVAL_HYBRID_W_DENSE", 0.60))
RETRIEVAL_HYBRID_W_LEXICAL = float(_cfg("RETRIEVAL_HYBRID_W_LEXICAL", 0.40))
RETRIEVAL_SECTION_RELEVANCE_BOOST = float(_cfg("RETRIEVAL_SECTION_RELEVANCE_BOOST", 0.06))
RETRIEVAL_CONTEXTUAL_HEADER_ENABLED = bool(_cfg("RETRIEVAL_CONTEXTUAL_HEADER_ENABLED", True))

QUERY_MIN_TERM_LEN = int(_cfg("QUERY_MIN_TERM_LEN", 2))
QUERY_STOPWORDS = set(_cfg("QUERY_STOPWORDS", []))

# --- V4 LLM Reranker config ---
V4_ENABLE_LLM_RERANK = bool(_cfg("V4_ENABLE_LLM_RERANK", True))  # Default: ON


def _contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def _normalize_lexical_text(text: str) -> str:
    text = normalize_whitespace(text or "")
    text = text.lower()
    text = LEXICAL_CLEAN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_text(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

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
    raw_tokens = _tokenize_text(text)
    filtered: list[str] = []

    for token in raw_tokens:
        if not token:
            continue
        if token in QUERY_STOPWORDS:
            continue
        if not _contains_cjk(token) and len(token) < QUERY_MIN_TERM_LEN:
            continue
        filtered.append(token)

    return filtered


def _normalize_mixed_language_query(text: str) -> str:
    text = normalize_whitespace(text or "")
    if not text:
        return ""

    text = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9]+)", r"\1 \2", text)
    text = re.sub(r"([A-Za-z0-9]+)([\u4e00-\u9fff])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_query(query: str) -> dict[str, Any]:
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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    if na == 0 or nb == 0:
        return 0.0

    return dot / (na * nb)


def _term_occurrence_detail(term: str, text: str) -> dict[str, Any]:
    if not term or not text:
        return {
            "score": 0.0,
            "exact_count": 0,
            "substring_count": 0,
            "whole_word": False,
            "prefix_hit": False,
        }

    normalized_text = text or ""
    lower_text = normalized_text.lower()
    lower_term = term.lower()

    exact_count = normalized_text.count(term)
    substring_count = lower_text.count(lower_term)

    whole_word = False
    prefix_hit = False

    if _contains_cjk(term):
        whole_word = exact_count > 0 or substring_count > 0
    else:
        word_pat = re.compile(rf"(?<![a-z0-9_]){re.escape(lower_term)}(?![a-z0-9_])")
        whole_word = bool(word_pat.search(lower_text))
        prefix_hit = bool(re.search(rf"(^|[\s:/._-]){re.escape(lower_term)}", lower_text))

    score = 0.0
    if exact_count > 0:
        score += min(3.0, exact_count * KEYWORD_EXACT_MATCH_WEIGHT)
    elif substring_count > 0:
        score += min(2.0, substring_count * KEYWORD_SUBSTRING_MATCH_WEIGHT)

    if whole_word:
        score += 0.35
    if prefix_hit:
        score += 0.2

    return {
        "score": score,
        "exact_count": exact_count,
        "substring_count": substring_count,
        "whole_word": whole_word,
        "prefix_hit": prefix_hit,
    }


def _safe_section_path(value: Any) -> str:
    if isinstance(value, list):
        return " / ".join(str(x).strip() for x in value if str(x).strip())
    if isinstance(value, dict):
        return normalize_whitespace(" ".join(str(v) for v in value.values()))
    return normalize_whitespace(str(value or ""))


def _row_to_candidate(row: Any) -> dict[str, Any]:
    metadata = safe_get(row, "metadata") or {}
    raw_section_path = safe_get(row, "section_path", "")

    lexical_text = normalize_whitespace(
        safe_get(row, "lexical_text", "")
        or safe_get(row, "search_text", "")
        or safe_get(row, "chunk_text", "")
    )
    search_text = normalize_whitespace(
        safe_get(row, "search_text", "")
        or safe_get(row, "chunk_text", "")
    )

    return {
        "chunk_id": safe_get(row, "id"),
        "document_id": safe_get(row, "document_id"),
        "chunk_index": safe_get(row, "chunk_index"),
        "chunk_text": normalize_whitespace(safe_get(row, "chunk_text", "")),
        "search_text": search_text,
        "lexical_text": lexical_text,
        "embedding": normalize_embedding(safe_get(row, "embedding")),
        "title": normalize_whitespace(
            safe_get(row, "title")
            or metadata.get("title", "")
        ),
        "section_path": _safe_section_path(raw_section_path),
        "section_title": normalize_whitespace(
            safe_get(row, "section_title")
            or metadata.get("section_title", "")
        ),
        "page_start": safe_get(row, "page_start"),
        "page_end": safe_get(row, "page_end"),
        "block_start_index": safe_get(row, "block_start_index"),
        "block_end_index": safe_get(row, "block_end_index"),
        "chunk_type": safe_get(row, "chunk_type"),
        "metadata": metadata,
        "embedding_score": 0.0,
        "keyword_score": 0.0,
        "bm25_score": 0.0,
        "title_match_score": 0.0,
        "section_match_score": 0.0,
        "coverage_score": 0.0,
        "matched_term_count": 0,
        "lexical_db_score": to_float(safe_get(row, "lexical_score")),
        "final_score": 0.0,
        "term_hits": {},
        "term_hit_detail": {},
        "is_neighbor": False,
    }


def _collect_row_texts(cand: dict[str, Any]) -> dict[str, str]:
    title = normalize_whitespace(cand.get("title", ""))
    section_title = normalize_whitespace(cand.get("section_title", ""))
    section_path = normalize_whitespace(cand.get("section_path", ""))
    search_text = normalize_whitespace(cand.get("search_text", ""))
    lexical_text = normalize_whitespace(cand.get("lexical_text", ""))

    return {
        "title": title,
        "section_title": section_title,
        "section_path": section_path,
        "section_text": normalize_whitespace(" ".join([section_path, section_title])),
        "search_text": search_text,
        "lexical_text": lexical_text,
    }


def _compute_keyword_components(query_info: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    terms = query_info.get("important_terms") or query_info.get("unique_terms") or []
    texts = _collect_row_texts(cand)

    keyword_raw = 0.0
    title_raw = 0.0
    section_raw = 0.0
    term_hits: Counter[str] = Counter()
    term_hit_detail: dict[str, dict[str, Any]] = {}

    for term in terms:
        body_hit = _term_occurrence_detail(term, texts["search_text"])
        if body_hit["score"] > 0:
            keyword_raw += body_hit["score"]
            term_hits[term] += body_hit["exact_count"] or body_hit["substring_count"] or 1
            term_hit_detail[term] = {
                "field": "search_text",
                **body_hit,
            }

        title_hit = _term_occurrence_detail(term, texts["title"])
        if title_hit["score"] > 0:
            title_raw += TITLE_MATCH_WEIGHT + (0.3 if title_hit["whole_word"] else 0.0)

        section_hit = _term_occurrence_detail(term, texts["section_text"])
        if section_hit["score"] > 0:
            section_raw += SECTION_MATCH_WEIGHT + (0.25 if section_hit["whole_word"] else 0.0)

    unique_term_count = max(1, len(terms))
    matched_term_count = len(term_hits)
    coverage_score = matched_term_count / unique_term_count

    keyword_score = min(1.0, (keyword_raw / 4.5) + coverage_score * 0.2)
    title_match_score = min(1.0, title_raw / 3.0)
    section_match_score = min(1.0, section_raw / 3.0)

    return {
        "keyword_score": keyword_score,
        "title_match_score": title_match_score,
        "section_match_score": section_match_score,
        "term_hits": dict(term_hits),
        "term_hit_detail": term_hit_detail,
        "matched_term_count": matched_term_count,
        "coverage_score": round(coverage_score, 6),
    }


def _normalize_scores(items: list[dict[str, Any]], field: str) -> None:
    max_score = max((to_float(item.get(field)) for item in items), default=0.0)
    if max_score <= 0:
        return
    for item in items:
        item[field] = to_float(item.get(field)) / max_score


def _hydrate_candidates(raw_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not raw_hits:
        return []

    by_id: dict[int, dict[str, Any]] = {}
    ordered_ids: list[int] = []

    for item in raw_hits:
        chunk_id = safe_get(item, "id")
        if chunk_id is None:
            chunk_id = safe_get(item, "chunk_id")
        if chunk_id is None:
            continue

        chunk_id = int(chunk_id)
        if chunk_id not in by_id:
            by_id[chunk_id] = dict(item)
            ordered_ids.append(chunk_id)
        else:
            existing = by_id[chunk_id]
            existing["lexical_score"] = max(
                to_float(existing.get("lexical_score")),
                to_float(item.get("lexical_score")),
            )

    hydrated_rows = get_chunks_by_ids(ordered_ids) or []
    row_map = {int(row["id"]): row for row in hydrated_rows if safe_get(row, "id") is not None}

    results: list[dict[str, Any]] = []
    for chunk_id in ordered_ids:
        row = row_map.get(chunk_id)
        if not row:
            continue
        cand = _row_to_candidate(row)
        extra = by_id.get(chunk_id, {})
        cand["lexical_db_score"] = max(
            to_float(cand.get("lexical_db_score")),
            to_float(extra.get("lexical_score")),
        )
        results.append(cand)

    return results


def _vector_recall_from_qdrant(
    query_info: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    query_text = query_info.get("normalized_query", "")
    if not query_text or top_k <= 0:
        return []

    hits = vector_store.search(query_text, top_k=top_k)
    if not hits:
        return []

    chunk_ids: list[int] = []
    for item in hits:
        chunk_id = item.get("chunk_id")
        if chunk_id is None:
            continue
        try:
            chunk_ids.append(int(chunk_id))
        except (TypeError, ValueError):
            continue

    if not chunk_ids:
        return []

    hydrated_rows = get_chunks_by_ids(chunk_ids) or []
    row_map = {int(row["id"]): row for row in hydrated_rows if safe_get(row, "id") is not None}

    results: list[dict[str, Any]] = []
    for hit in hits:
        chunk_id = hit.get("chunk_id")
        if chunk_id is None:
            continue

        try:
            chunk_id = int(chunk_id)
        except (TypeError, ValueError):
            continue

        row = row_map.get(chunk_id)
        if not row:
            continue

        cand = _row_to_candidate(row)
        cand["embedding_score"] = to_float(hit.get("embedding_score"))
        cand["final_score"] = cand["embedding_score"]

        if hit.get("document_id") is not None:
            cand["document_id"] = hit.get("document_id")
        if hit.get("chunk_index") is not None:
            cand["chunk_index"] = hit.get("chunk_index")

        results.append(cand)

    _normalize_scores(results, "embedding_score")
    for item in results:
        item["final_score"] = to_float(item.get("embedding_score"))

    results.sort(key=lambda x: x.get("embedding_score", 0.0), reverse=True)
    return results[:top_k]


def _keyword_recall_from_candidates(
    query_info: dict[str, Any],
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for cand in candidates:
        item = dict(cand)
        item.update(_compute_keyword_components(query_info, item))
        lexical_score = (
            0.60 * to_float(item.get("keyword_score"))
            + 0.23 * to_float(item.get("title_match_score"))
            + 0.17 * to_float(item.get("section_match_score"))
        )
        item["final_score"] = lexical_score
        if lexical_score > 0:
            results.append(item)

    results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return results[:top_k]


def _lexical_recall_from_db(
    query_info: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    query_text = query_info.get("normalized_query", "")
    if not query_text:
        return []

    natural_hits = search_chunks_fulltext(
        query_text,
        limit=max(top_k, HYBRID_LEXICAL_FETCH_K),
    ) or []

    boolean_hits = search_chunks_boolean(
        query_text,
        limit=max(top_k, HYBRID_BOOLEAN_FETCH_K),
        require_all_terms=False,
    ) or []

    merged_raw: dict[int, dict[str, Any]] = {}
    for row in [*natural_hits, *boolean_hits]:
        chunk_id = safe_get(row, "id")
        if chunk_id is None:
            continue
        chunk_id = int(chunk_id)
        if chunk_id not in merged_raw:
            merged_raw[chunk_id] = dict(row)
        else:
            merged_raw[chunk_id]["lexical_score"] = max(
                to_float(merged_raw[chunk_id].get("lexical_score")),
                to_float(row.get("lexical_score")),
            )

    hydrated = _hydrate_candidates(list(merged_raw.values()))
    _normalize_scores(hydrated, "lexical_db_score")

    for cand in hydrated:
        cand["bm25_score"] = to_float(cand.get("lexical_db_score"))

    hydrated.sort(key=lambda x: x.get("bm25_score", 0.0), reverse=True)
    return hydrated[:top_k]


def _merge_recall_candidates(*candidate_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[Any, dict[str, Any]] = {}

    def merge_one(item: dict[str, Any]) -> None:
        chunk_id = item.get("chunk_id")
        if chunk_id not in merged:
            merged[chunk_id] = dict(item)
            return

        current = merged[chunk_id]

        for field in [
            "embedding_score",
            "keyword_score",
            "bm25_score",
            "lexical_db_score",
            "title_match_score",
            "section_match_score",
            "coverage_score",
            "matched_term_count",
            "final_score",
        ]:
            current[field] = max(to_float(current.get(field)), to_float(item.get(field)))

        if item.get("term_hits"):
            existing_hits = Counter(current.get("term_hits") or {})
            existing_hits.update(item.get("term_hits") or {})
            current["term_hits"] = dict(existing_hits)

        if item.get("term_hit_detail"):
            details = dict(current.get("term_hit_detail") or {})
            details.update(item.get("term_hit_detail") or {})
            current["term_hit_detail"] = details

    for group in candidate_groups:
        for item in group:
            merge_one(item)

    return list(merged.values())


def _length_penalty(text: str) -> float:
    text_len = len(text or "")
    if text_len < 30:
        return 0.05
    if text_len < 60:
        return 0.02
    if text_len > 1800:
        return 0.10
    if text_len > 1200:
        return 0.06
    return 0.0


def _metadata_bonus(cand: dict[str, Any]) -> float:
    metadata = cand.get("metadata") or {}

    bonus = 0.0

    block_count = safe_get(metadata, "block_count", 0) or 0
    heading_depth = safe_get(metadata, "heading_path_depth", 0) or 0
    has_list = bool(safe_get(metadata, "has_list", False))
    has_table = bool(safe_get(metadata, "has_table", False))

    if 1 <= to_float(block_count) <= 6:
        bonus += 0.02
    if 1 <= to_float(heading_depth) <= 4:
        bonus += 0.015
    if has_list:
        bonus += 0.01
    if has_table:
        bonus += 0.008

    return bonus


# =============================================================================
# Query Intent Classification
# =============================================================================

# Query intent types
QUERY_INTENT_NUMERIC = "numeric_fact"
QUERY_INTENT_LIST = "list_fact"
QUERY_INTENT_DESCRIPTIVE = "descriptive"
QUERY_INTENT_HYBRID = "hybrid"

# Patterns for numeric_fact queries
_NUMERIC_FACT_PATTER = re.compile(
    r"\b(ratio|tax rate|effective tax|inventory|cash|revenue|profit|margin|"
    r"earnings|eps|ebitda|debt|liabilities|assets|current|liquidity|"
    r"dividend|percentage|%|percent|million|billion|amount|how much|how many|"
    r"total|cost|price|value|worth|balance|decline|increase|growth|driven)\b",
    re.IGNORECASE,
)

# Patterns for list_fact queries
_LIST_FACT_PATTER = re.compile(
    r"\b(products|services|segments|subsidiaries|brands|offerings|"
    r"what does .* offer|what are .* products|what .* provide|"
    r"types of|categories|classifications|lines of business|"
    r"business.* overview|operating.* segment|revenue.* segment)\b",
    re.IGNORECASE,
)

# Patterns for descriptive queries
_DESCRIPTIVE_PATTER = re.compile(
    r"\b(explain|discuss|describe|why|how did|how does|what is the reason|"
    r"what caused|management.* discussion|overview.* business|"
    r"trend|comparison|year over year|vs |versus)\b",
    re.IGNORECASE,
)


def classify_query_intent(query: str) -> str:
    """
    Classify query into intent types:
      - numeric_fact  : query asks for a specific number/ratio/amount
      - list_fact     : query asks for items/products/services/segments
      - descriptive   : query asks for explanation/discussion
      - hybrid        : combination or unclear

    This is used to bias reranking toward the right content type.
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


# =============================================================================
# Per-chunk Signal Functions
# =============================================================================

# Financial table-indicator keywords (high value when query is about numeric facts)
_TABLE_STRUCTURE_KEYWORDS = [
    "total", "assets", "liabilities", "equity", "inventory", "revenue",
    "tax", "rate", "cash", "current", "net income", "operating",
    "diluted", "shares", "eps", "dividend", "debt", "gross", "margin",
    "cost of sales", "operating income", "pretax", "provision",
    "balance sheet", "income statement", "cash flow",
    "current assets", "current liabilities", "long-term",
    "accounts receivable", "accounts payable", "property",
]
# Noise section keywords (penalize these when query is numeric/list)
_NOISE_SECTION_PATTERNS = [
    r"table of contents", r"toc", r"sec filing", r"form 10-k",
    r"forward-looking", r"risk factor", r"item [0-9]",
    r"signature page", r"exhibit index", r"index of exhibits",
    r"schedule .{0,30}", r"attachment .{0,30}",
]


def _compute_numeric_density_boost(cand: dict[str, Any]) -> float:
    """
    Boost chunks with high numeric density.
    Returns a bonus in [0.0, RETRIEVAL_NUMERIC_BOOST_WEIGHT].
    """
    text = (cand.get("chunk_text", "") or "") + " " + (cand.get("search_text", "") or "")
    dollar_count = text.count("$")
    pct_count = text.count("%")
    number_count = len(re.findall(r"[\d,]+\.?\d*", text))

    # Count financial number patterns
    fin_pattern_matches = 0
    for pat in _NUMERIC_FINANCE_PATTERNS:
        try:
            fin_pattern_matches += len(re.findall(pat, text))
        except re.error:
            pass

    # A "dense" numeric chunk has at least 2 of: dollar signs, percents, financial patterns
    signals = sum([
        1 if dollar_count >= 1 else 0,
        1 if pct_count >= 1 else 0,
        1 if fin_pattern_matches >= 2 else 0,
        1 if number_count >= 5 else 0,
    ])

    if signals >= 3:
        return RETRIEVAL_NUMERIC_BOOST_WEIGHT
    elif signals == 2:
        return RETRIEVAL_NUMERIC_BOOST_WEIGHT * 0.6
    elif signals == 1:
        return RETRIEVAL_NUMERIC_BOOST_WEIGHT * 0.25
    return 0.0


def _compute_table_like_boost(cand: dict[str, Any]) -> float:
    """
    Boost chunks that look like financial table slices.
    Heuristics: many rows with numbers, presence of table header keywords.
    """
    text = (cand.get("chunk_text", "") or "").lower()
    section = ((cand.get("section_title", "") or "") + " " +
               (cand.get("section_path", "") or "")).lower()

    # Section title checks
    table_title_hits = sum(1 for kw in _TABLE_STRUCTURE_KEYWORDS if kw in section)
    if table_title_hits >= 2:
        title_bonus = 0.03
    elif table_title_hits == 1:
        title_bonus = 0.015
    else:
        title_bonus = 0.0

    # Text structure checks (many lines with numbers = likely table)
    lines = text.split("\n")
    numeric_lines = sum(1 for line in lines if re.search(r"[\d,]+\.?\d*", line))
    if len(lines) >= 3 and numeric_lines / len(lines) >= 0.4:
        structure_bonus = RETRIEVAL_TABLE_BOOST_WEIGHT * 0.7
    elif numeric_lines >= 2:
        structure_bonus = RETRIEVAL_TABLE_BOOST_WEIGHT * 0.35
    else:
        structure_bonus = 0.0

    return min(RETRIEVAL_TABLE_BOOST_WEIGHT, title_bonus + structure_bonus)


def _compute_query_aware_lexical_boost(
    cand: dict[str, Any],
    query_info: dict[str, Any],
    intent: str,
) -> float:
    """
    Boost chunks based on query intent and chunk content alignment.
    E.g., for numeric_fact queries, boost balance-sheet/table chunks;
          for list_fact queries, boost overview/products/segment chunks.
    """
    text_lower = ((cand.get("chunk_text", "") or "") + " " +
                  (cand.get("search_text", "") or "")).lower()
    section_lower = ((cand.get("section_title", "") or "") + " " +
                     (cand.get("section_path", "") or "")).lower()
    combined = text_lower + " " + section_lower

    if intent == QUERY_INTENT_NUMERIC:
        # Boost balance sheet, financial summary, ratio-related content
        numeric_section_kw = [
            "balance sheet", "current assets", "current liabilities",
            "cash equivalents", "short-term investment", "accounts receivable",
            "inventori", "total current", "financial summary",
            "ratio", "quick ratio", "current ratio",
            "effective tax", "provision for income",
            "income tax", "tax rate", "operating income",
            "net income", "gross margin", "revenue",
        ]
        hits = sum(1 for kw in numeric_section_kw if kw in combined)
        if hits >= 3:
            return RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT
        elif hits == 2:
            return RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT * 0.65
        elif hits == 1:
            return RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT * 0.3
        return 0.0

    elif intent == QUERY_INTENT_LIST:
        # Boost overview, products, segments, business description
        list_section_kw = [
            "overview", "products", "services", "segment", "subsidiary",
            "brands", "business", "offering", "we offer", "our products",
            "our business", "principal products", "line of business",
            "geographic", "category",
        ]
        hits = sum(1 for kw in list_section_kw if kw in combined)
        if hits >= 2:
            return RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT
        elif hits == 1:
            return RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT * 0.5
        return 0.0

    elif intent == QUERY_INTENT_DESCRIPTIVE:
        # For descriptive queries, slightly boost narrative sections
        narrative_kw = [
            "management", "discussion", "analysis", "overview",
            "liquidity", "capital resources", "results of operations",
        ]
        hits = sum(1 for kw in narrative_kw if kw in section_lower)
        if hits >= 1:
            return RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT * 0.5
        return 0.0

    return 0.0  # hybrid or unknown


def _compute_anti_noise_penalty(cand: dict[str, Any]) -> float:
    """
    Mildly penalize chunks from noise sections (TOC, risk factors, etc.)
    that are unlikely to contain answer-worthy content.
    """
    section = ((cand.get("section_title", "") or "") + " " +
               (cand.get("section_path", "") or "")).lower()

    for pat in _NOISE_SECTION_PATTERNS:
        if re.search(pat, section):
            return -RETRIEVAL_ANTI_NOISE_PENALTY_WEIGHT

    # Also penalize very short chunks with no numbers (likely filler)
    text = (cand.get("chunk_text", "") or "")
    if len(text) < 80 and not re.search(r"\d", text):
        return -RETRIEVAL_ANTI_NOISE_PENALTY_WEIGHT * 0.5

    return 0.0


def _smooth_page_scores(
    candidates: list[dict[str, Any]],
    score_key: str = "final_score",
    alpha: float | None = None,
) -> dict[int, float]:
    """
    Page-level neighborhood smoothing: for each page p, compute
      smoothed_score(p) = score(p) + alpha * max(score(neighbor pages))

    This promotes clusters of high-scoring adjacent pages, helping
    convert relaxed hits into strict hits.

    Returns a dict {page: smoothed_score}.
    """
    if alpha is None:
        alpha = RETRIEVAL_PAGE_CLUSTER_ALPHA

    # Collect base scores per page
    page_base: dict[int, float] = {}
    for cand in candidates:
        p = cand.get("page_start")
        if p is None:
            continue
        try:
            page = int(p)
        except (TypeError, ValueError):
            continue
        score = to_float(cand.get(score_key, 0.0))
        if page not in page_base or score > page_base[page]:
            page_base[page] = score

    if not page_base:
        return {}

    smoothed: dict[int, float] = {}
    for page, base_score in page_base.items():
        neighbor_max = 0.0
        for neighbor in (page - 1, page + 1):
            neighbor_score = page_base.get(neighbor, 0.0)
            if neighbor_score > neighbor_max:
                neighbor_max = neighbor_score
        smoothed[page] = base_score + alpha * neighbor_max

    return smoothed


# Narrative sections in 10-K that tend to contain analytical content vs raw tables
_NARRATIVE_SECTION_KEYWORDS = [
    "management", "discussion", "analysis",
    "liquidity", "capital resources", "financial condition",
    "results of operations", "overview",
    "risk factor", "business",
    "notes to consolidated", "note ",  # footnotes / notes are narrative
]
# Sections that are pure data tables - lower narrative value
_NOISE_SECTION_KEYWORDS = [
    "schedule", "form 10-k", "table of contents",
    "sec filing", "accession",
]
# Financial query terms that, when present in a chunk, signal high relevance
_FINANCE_TERM_KEYWORDS = [
    "ratio", "revenue", "cash", "debt", "eps", "ebitda",
    "inventory", "capex", "depreciation", "amortization",
    "segment", "gross margin", "operating income", "net income",
    "assets", "liabilities", "equity", "dividend",
    "growth", "decline", "increase", "decrease", "driven",
    "liquidity", "solvency", "working capital",
]


def _section_narrative_bonus(cand: dict[str, Any]) -> float:
    """Boost chunks from narrative/explanatory sections vs pure tables/indices."""
    section = cand.get("section_title", "") or ""
    section_lower = section.lower()
    path = cand.get("section_path", "") or ""
    path_lower = path.lower()
    chunk_text = (cand.get("chunk_text", "") or "").lower()
    search_text = (cand.get("search_text", "") or "").lower()
    combined = section_lower + " " + path_lower

    # Penalize noise sections
    for noise in _NOISE_SECTION_KEYWORDS:
        if noise in combined:
            return -0.05

    # Boost narrative sections
    narrative_score = 0.0
    for narrative in _NARRATIVE_SECTION_KEYWORDS:
        if narrative in combined:
            narrative_score = 0.06
            break

    # Additional boost if chunk text also has finance terms
    combined_text = chunk_text + " " + search_text
    finance_hits = sum(1 for kw in _FINANCE_TERM_KEYWORDS if kw in combined_text)
    if finance_hits >= 3:
        narrative_score += 0.05
    elif finance_hits >= 1:
        narrative_score += 0.025

    return narrative_score


# Financial causal/phrasal patterns that signal explanatory content
_CAUSAL_PHRASE_PATTERNS = [
    r"driven by",
    r"primarily driven by",
    r"mainly driven by",
    r"chiefly driven by",
    r"primarily due to",
    r"mainly due to",
    r"chiefly due to",
    r"attributable to",
    r"explained by",
    r"resulted from",
    r"mainly because",
    r"primarily because",
    r"arising from",
    r"due primarily to",
    r"due mainly to",
    r"due in part to",
    r"reflected in",
    r"reflecting",
    r"primarily reflects",
    r"mainly reflects",
]

# Numeric financial content indicators (lighter patterns)
_NUMERIC_FINANCE_PATTERNS = [
    r"\$[\d,]+",          # Dollar amounts
    r"\$\d+\.\d+",        # Dollars with cents
    r"[\d,]+\.?\d*%",     # Percentages
    r"\d+%",              # Percentages
    r"\$\d+ billion",
    r"\$\d+ million",
    r"\d+ million",
    r"\d+ billion",
    r"(margin|gross|operating|net|EBITDA|EPS)[^\.]{0,100}\$[\d,]+",
]


def _compute_financial_content_bonus(
    cand: dict[str, Any],
    query_info: dict[str, Any],
) -> float:
    """
    Boost chunks that contain financial explanatory content and numeric detail.
    Combines causal phrase detection with numeric density weighting.
    """
    text = (cand.get("chunk_text", "") or "").lower()
    search_text = (cand.get("search_text", "") or "").lower()
    combined = text + " " + search_text

    # 1. Causal phrase bonus
    causal_count = 0
    for pattern in _CAUSAL_PHRASE_PATTERNS:
        if pattern in combined:
            causal_count += 1
    causal_bonus = 0.0
    if causal_count >= 1:
        causal_bonus = min(0.06, 0.025 * causal_count)

    # 2. Numeric density bonus: chunks with financial numbers are generally more informative
    # Count dollar/percent patterns
    dollar_signs = text.count("$")
    pct_signs = text.count("%")
    # Count specific financial number patterns
    numeric_matches = 0
    for pattern in _NUMERIC_FINANCE_PATTERNS:
        try:
            numeric_matches += len(re.findall(pattern, combined))
        except re.error:
            pass

    # Chunk with both numbers AND causal phrases is especially valuable
    has_numbers = dollar_signs >= 1 or pct_signs >= 1
    has_causal = causal_count >= 1
    numeric_bonus = 0.0
    if has_numbers and has_causal:
        numeric_bonus = 0.04  # Gold: explanatory + numeric
    elif has_causal:
        numeric_bonus = 0.02  # Explanatory without numbers

    # 3. Query-aware financial term boost
    # If query is about drivers/causes, boost explanatory chunks
    query_lower = query_info.get("normalized_query", "").lower()
    is_driver_query = any(
        kw in query_lower for kw in ["driven", "driver", "cause", "why", "due to", "reason", "change", "increase", "decrease", "growth", "decline"]
    )
    if is_driver_query and causal_count >= 1:
        causal_bonus *= 1.5  # 1.5x boost for driver queries with explanatory chunks

    return causal_bonus + numeric_bonus


def _compute_page_diversity_bonus(
    candidates: list[dict[str, Any]],
    page_counts: dict[int, int],
    top_n: int = 20,
) -> dict[int, float]:
    """
    Compute a page-level diversity bonus.
    Pages that are under-represented in the top-N candidates get a boost
    to promote coverage of multiple pages.
    """
    bonus: dict[int, float] = {}
    if not candidates or not page_counts:
        return bonus

    top_candidates = candidates[:top_n]
    # How many different pages in top-N
    top_pages = set()
    for c in top_candidates:
        p = c.get("page_start")
        if p is not None:
            try:
                top_pages.add(int(p))
            except (TypeError, ValueError):
                pass

    total_pages = len(page_counts)
    if total_pages <= 1:
        return bonus

    # Ideal: each page appears at most once in top-N (diverse pages)
    # Penalty for over-represented pages, bonus for under-represented
    max_pages_desired = min(total_pages, 5)  # aim for up to 5 diverse pages

    for cand in candidates:
        p = cand.get("page_start")
        if p is None:
            continue
        try:
            page = int(p)
        except (TypeError, ValueError):
            continue

        count_in_top = page_counts.get(page, 0)

        # Pages with 0 count in top get bonus, pages with 1+ get decreasing bonus
        if count_in_top == 0:
            bonus[page] = 0.10
        elif count_in_top == 1:
            bonus[page] = 0.04

    return bonus


# Financial-domain query terms that signal a need for secondary balance-sheet recall
_FINANCIAL_QUERY_INDICATORS = [
    "ratio", "liquidity", "solvency", "current ratio", "quick ratio",
    "balance sheet", "current assets", "current liabilities",
    "working capital", "cash position", "debt", "leverage",
    "revenue", "income", "profit", "margin", "eps",
    "segment", "growth", "decline",
]
# Additional terms to add to the query when financial indicators are detected
_FINANCIAL_QUERY_EXPANSION_TERMS = [
    "current assets", "current liabilities", "cash equivalents",
    "short-term investments", "balance sheet", "liquidity",
    "financial condition", "working capital",
]


def _detect_financial_query(query_info: dict[str, Any]) -> bool:
    """Return True if the query likely asks about financial metrics/ratios."""
    query_lower = query_info.get("normalized_query", "").lower()
    # Only detect if the ORIGINAL query itself contains financial terms
    # (not just any query triggering the secondary recall)
    for indicator in _FINANCIAL_QUERY_INDICATORS:
        if indicator in query_lower:
            return True
    return False


def _query_has_financial_terms(query_info: dict[str, Any]) -> bool:
    """Return True if the query's important/unique terms include financial vocabulary."""
    terms = query_info.get("important_terms", []) or query_info.get("unique_terms", [])
    terms_lower = " ".join(t.lower() for t in terms)
    for indicator in _FINANCIAL_QUERY_INDICATORS:
        if indicator in terms_lower:
            return True
    return False


def _build_expanded_query(query_info: dict[str, Any]) -> str:
    """
    Build an expanded query for secondary lexical recall on financial topics.
    Only adds expansion terms if the original query contains financial vocabulary.
    """
    # Only expand if the query itself has financial terms
    if not _query_has_financial_terms(query_info):
        return ""

    base = query_info.get("normalized_query", "")
    terms = query_info.get("important_terms", []) or query_info.get("unique_terms", [])
    # Add financial-domain terms that aren't already in the query
    terms_lower = {t.lower() for t in terms}
    expanded_parts = [base]
    for term in _FINANCIAL_QUERY_EXPANSION_TERMS:
        if term.lower() not in terms_lower:
            expanded_parts.append(term)
    return " ".join(expanded_parts)


def _secondary_financial_recall(
    query_info: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Perform a secondary lexical recall specifically for financial statement content.
    Only activated when _detect_financial_query returns True.
    """
    if not _detect_financial_query(query_info):
        return []

    expanded_query = _build_expanded_query(query_info)
    if not expanded_query.strip():
        return []

    # Use the full-text search for the expanded query
    financial_hits = search_chunks_fulltext(
        expanded_query,
        limit=max(top_k, 60),
    ) or []

    # Also try boolean search with OR of key terms
    boolean_hits = search_chunks_boolean(
        expanded_query,
        limit=max(top_k, 40),
        require_all_terms=False,
    ) or []

    # Merge deduplicated by id
    seen_ids: set[int] = set()
    merged: list[dict[str, Any]] = []
    for row in [*financial_hits, *boolean_hits]:
        chunk_id = safe_get(row, "id")
        if chunk_id is None:
            continue
        chunk_id = int(chunk_id)
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            merged.append(dict(row))

    # Hydrate and convert to candidates
    candidates = _hydrate_candidates(merged)
    _normalize_scores(candidates, "lexical_db_score")
    for cand in candidates:
        cand["bm25_score"] = to_float(cand.get("lexical_db_score"))

    candidates.sort(key=lambda x: x.get("bm25_score", 0.0), reverse=True)
    return candidates[:top_k]


def _rerank_hybrid_candidates(
    candidates: list[dict[str, Any]],
    query_info: dict[str, Any],
    enable_page_diversity: bool = True,
    enable_page_clustering: bool = True,
) -> list[dict[str, Any]]:
    """
    Rerank hybrid candidates using weighted signals + new rerank signals.

    New signals (P0):
      - numeric_density_boost   : chunk has financial numbers/density
      - table_like_boost        : chunk looks like a financial table
      - query_aware_lexical_boost: intent-based lexical alignment
      - anti_noise_penalty       : penalize TOC/risk-factor/filler chunks
      - page_clustering          : neighborhood smoothing for page scores
    """
    if not candidates:
        return []

    # Classify query intent once
    raw_query = query_info.get("raw_query", "")
    intent = classify_query_intent(raw_query)

    reranked: list[dict[str, Any]] = []
    query_terms = query_info.get("important_terms") or query_info.get("unique_terms") or []

    # Compute page counts across ALL candidates (before reranking)
    all_page_counts: dict[int, int] = {}
    for cand in candidates:
        p = cand.get("page_start")
        if p is not None:
            try:
                page = int(p)
                all_page_counts[page] = all_page_counts.get(page, 0) + 1
            except (TypeError, ValueError):
                pass

    page_diversity_bonus: dict[int, float] = {}
    if enable_page_diversity and len(all_page_counts) > 1:
        page_diversity_bonus = _compute_page_diversity_bonus(candidates, all_page_counts)

    # Page clustering: compute smoothed page scores for rerank
    # We compute initial scores first, then apply smoothing
    # to avoid biasing the initial ranking
    page_cluster_scores: dict[int, float] = {}

    for cand in candidates:
        text = cand.get("search_text", "") or cand.get("chunk_text", "")

        embedding_score = to_float(cand.get("embedding_score"))
        keyword_score = to_float(cand.get("keyword_score"))
        bm25_score = to_float(cand.get("bm25_score"))
        title_score = to_float(cand.get("title_match_score"))
        section_score = to_float(cand.get("section_match_score"))
        coverage_score = to_float(cand.get("coverage_score"))

        # For candidates that only have bm25 (no embedding/keyword scores),
        # compute keyword components on-the-fly to give them a fair score
        has_keyword_scores = keyword_score > 0 or coverage_score > 0
        if not has_keyword_scores and bm25_score > 0 and text:
            kw_comps = _compute_keyword_components(query_info, cand)
            keyword_score = to_float(kw_comps.get("keyword_score"))
            title_score = to_float(kw_comps.get("title_match_score"))
            section_score = to_float(kw_comps.get("section_match_score"))
            coverage_score = to_float(kw_comps.get("coverage_score", 0))

        coverage_bonus = 0.08 * coverage_score if query_terms else 0.0

        if len(query_terms) <= 2:
            embedding_weight = max(0.0, RETRIEVAL_WEIGHT_EMBEDDING - 0.05)
            keyword_weight = RETRIEVAL_WEIGHT_KEYWORD + 0.03
            bm25_weight = RETRIEVAL_WEIGHT_BM25 + 0.02
        else:
            embedding_weight = RETRIEVAL_WEIGHT_EMBEDDING
            keyword_weight = RETRIEVAL_WEIGHT_KEYWORD
            bm25_weight = RETRIEVAL_WEIGHT_BM25

        # Special boost for lexical-only secondary candidates (high bm25, no embedding):
        is_secondary_only = embedding_score == 0 and bm25_score > 0
        secondary_bonus = 0.0
        if is_secondary_only:
            keyword_weight = 0.35
            bm25_weight = 0.25
            secondary_bonus = 0.06

        # Page diversity bonus
        p = cand.get("page_start")
        page_div = 0.0
        if p is not None:
            try:
                page_div = page_diversity_bonus.get(int(p), 0.0)
            except (TypeError, ValueError):
                pass

        # Financial content bonus: causal phrases + numeric density (existing)
        financial_bonus = _compute_financial_content_bonus(cand, query_info)

        # --- New P0 signals ---
        numeric_boost = _compute_numeric_density_boost(cand)
        table_boost = _compute_table_like_boost(cand)
        query_aware_boost = _compute_query_aware_lexical_boost(cand, query_info, intent)
        anti_noise_penalty = _compute_anti_noise_penalty(cand)

        base_score = (
            embedding_weight * embedding_score
            + keyword_weight * keyword_score
            + bm25_weight * bm25_score
            + RETRIEVAL_WEIGHT_TITLE * title_score
            + RETRIEVAL_WEIGHT_SECTION * section_score
            + coverage_bonus
            + _metadata_bonus(cand)
            + _section_narrative_bonus(cand)
            + page_div
            + secondary_bonus
            + financial_bonus
            + numeric_boost
            + table_boost
            + query_aware_boost
            + anti_noise_penalty
            - _length_penalty(text)
        )

        item = dict(cand)
        item["final_score"] = base_score
        item["_numeric_boost"] = round(numeric_boost, 6)
        item["_table_boost"] = round(table_boost, 6)
        item["_query_aware_boost"] = round(query_aware_boost, 6)
        item["_anti_noise_penalty"] = round(anti_noise_penalty, 6)
        item["_query_intent"] = intent
        reranked.append(item)

    reranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    # Apply page clustering as a post-pass: re-sort by cluster-boosted score
    if enable_page_clustering and len(reranked) > 1:
        # First, get smoothed scores using initial ranking as base
        smoothed = _smooth_page_scores(reranked, score_key="final_score")
        if smoothed:
            # For each page, find the best-ranked chunk and boost its score
            # by the cluster bonus; then re-sort
            best_per_page: dict[int, dict[str, Any]] = {}
            for cand in reranked:
                p = cand.get("page_start")
                if p is None:
                    continue
                try:
                    page = int(p)
                except (TypeError, ValueError):
                    continue
                if page not in best_per_page:
                    best_per_page[page] = cand
                elif cand.get("final_score", 0) > best_per_page[page].get("final_score", 0):
                    best_per_page[page] = cand

            # Apply cluster bonus to best-per-page candidates
            for page, cand in best_per_page.items():
                cluster_bonus = smoothed.get(page, 0.0) - cand.get("final_score", 0.0)
                if cluster_bonus > 0.01:
                    cand["final_score"] += cluster_bonus
                    cand["_page_cluster_bonus"] = round(cluster_bonus, 6)

            reranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    return reranked


def _text_similarity_for_dedup(a: str, b: str) -> float:
    tokens_a = set(_tokenize_text(_normalize_lexical_text(a)))
    tokens_b = set(_tokenize_text(_normalize_lexical_text(b)))

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))


def _deduplicate_candidates(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    section_counter: defaultdict[str, int] = defaultdict(int)

    for cand in candidates:
        section_key = f"{cand.get('document_id')}::{cand.get('section_path') or cand.get('section_title') or ''}"
        if section_counter[section_key] >= RETRIEVAL_MAX_SAME_SECTION:
            continue

        is_dup = False
        for chosen in selected:
            sim = _text_similarity_for_dedup(
                cand.get("search_text", "") or cand.get("chunk_text", ""),
                chosen.get("search_text", "") or chosen.get("chunk_text", ""),
            )
            if sim >= RETRIEVAL_DEDUP_SIM_THRESHOLD:
                is_dup = True
                break

        if is_dup:
            continue

        selected.append(cand)
        section_counter[section_key] += 1

        if len(selected) >= top_k:
            break

    return selected


# Max chunks per page in the final result before page-diversity cap
MAX_CHUNKS_PER_PAGE = int(_cfg("RETRIEVAL_MAX_CHUNKS_PER_PAGE", 2))


def _cap_page_duplicates(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """
    Apply a hard cap on how many chunks from the same page can appear in top_k.
    Keeps the highest-scoring chunk per page up to cap, then fills remaining
    slots with the best chunks from other (under-represented) pages.
    """
    if not candidates:
        return []

    page_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        p = cand.get("page_start")
        if p is not None:
            try:
                page_groups[int(p)].append(cand)
            except (TypeError, ValueError):
                page_groups[id(cand)].append(cand)  # group by object id for None pages

    # Sort chunks within each page by score (descending)
    for page in page_groups:
        page_groups[page].sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    selected: list[dict[str, Any]] = []
    surplus: list[dict[str, Any]] = []  # lower-ranked from over-represented pages

    # First pass: select top chunks from each page, up to MAX_PER_PAGE
    for page, chunks in page_groups.items():
        selected.extend(chunks[:MAX_CHUNKS_PER_PAGE])
        surplus.extend(chunks[MAX_CHUNKS_PER_PAGE:])

    # Sort selected by score descending
    selected.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    # Second pass: fill remaining slots with best surplus chunks (diverse pages first)
    surplus.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    selected.extend(surplus[: max(0, top_k - len(selected))])

    # Final sort by score
    selected.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return selected[:top_k]


def _expand_neighbor_chunks(candidates: list[dict[str, Any]], target_limit: int) -> list[dict[str, Any]]:
    if not RETRIEVAL_ENABLE_NEIGHBOR_EXPANSION or not candidates:
        return candidates[:target_limit]

    results: list[dict[str, Any]] = []
    seen_ids = set()
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for cand in candidates:
        doc_id = cand.get("document_id")
        if doc_id is not None:
            grouped[int(doc_id)].append(cand)

    for doc_id, doc_cands in grouped.items():
        center_indexes = [
            int(c.get("chunk_index"))
            for c in doc_cands
            if c.get("chunk_index") is not None
        ]
        neighbor_rows = get_neighbor_chunks(
            document_id=doc_id,
            center_chunk_indexes=center_indexes,
            window=RETRIEVAL_NEIGHBOR_WINDOW,
        ) or []

        index_map = {safe_get(r, "chunk_index"): r for r in neighbor_rows}

        for cand in doc_cands:
            cid = cand.get("chunk_id")
            if cid not in seen_ids:
                results.append(cand)
                seen_ids.add(cid)

            center_index = cand.get("chunk_index")
            center_section = cand.get("section_path") or cand.get("section_title") or ""
            center_score = to_float(cand.get("final_score"))

            if center_index is None:
                continue

            for offset in range(1, RETRIEVAL_NEIGHBOR_WINDOW + 1):
                for neighbor_idx in (center_index - offset, center_index + offset):
                    row = index_map.get(neighbor_idx)
                    if not row:
                        continue

                    neighbor_id = safe_get(row, "id")
                    if neighbor_id in seen_ids:
                        continue

                    neighbor_cand = _row_to_candidate(row)
                    neighbor_section = neighbor_cand.get("section_path") or neighbor_cand.get("section_title") or ""

                    if center_section and neighbor_section and center_section != neighbor_section:
                        continue

                    neighbor_cand["final_score"] = center_score * (0.90 - (offset - 1) * 0.04)
                    neighbor_cand["is_neighbor"] = True
                    # Propagate query intent and rerank signals from center to neighbor
                    neighbor_cand["_query_intent"] = cand.get("_query_intent", "unknown")
                    neighbor_cand["_numeric_boost"] = cand.get("_numeric_boost", 0.0)
                    neighbor_cand["_table_boost"] = cand.get("_table_boost", 0.0)
                    neighbor_cand["_query_aware_boost"] = cand.get("_query_aware_boost", 0.0)
                    neighbor_cand["_anti_noise_penalty"] = cand.get("_anti_noise_penalty", 0.0)

                    results.append(neighbor_cand)
                    seen_ids.add(neighbor_id)

                    if len(results) >= target_limit:
                        return results[:target_limit]

    return results[:target_limit]


def enhance_financial_query(query: str) -> str:
    """
    Lightweight query enhancement for financial document retrieval.
    Appends domain-specific terms based on detected query intent.
    Does NOT rewrite the query - only augments it for retrieval.
    """
    query_lower = query.lower()

    # Detect query intent and add relevant financial terms
    enhanced_parts = [query]

    # Ratio / liquidity / balance sheet queries
    if any(kw in query_lower for kw in ["ratio", "liquidity", "quick ratio", "current ratio"]):
        enhanced_parts.extend([
            "current assets current liabilities",
            "cash equivalents short-term investments",
            "balance sheet financial condition",
            "accounts receivable inventories",
        ])

    # Revenue / sales queries
    if any(kw in query_lower for kw in ["revenue", "sales", "net sales", "income"]):
        enhanced_parts.extend([
            "net sales revenue growth decline",
            "cost of sales gross margin",
            "operating income segment",
        ])

    # Cash flow queries
    if any(kw in query_lower for kw in ["cash", "cash flow", "liquidity"]):
        enhanced_parts.extend([
            "cash flow operating investing financing",
            "cash equivalents liquidity",
            "operating cash flow",
        ])

    # Debt / liabilities queries
    if any(kw in query_lower for kw in ["debt", "liabilities", "borrowing", "leverage", "leverage ratio"]):
        enhanced_parts.extend([
            "debt liabilities borrowings",
            "long-term debt current liabilities",
            "financial condition obligations",
        ])

    # Segment / business queries
    if any(kw in query_lower for kw in ["segment", "business segment", "operating income segment"]):
        enhanced_parts.extend([
            "segment business operating income",
            "revenue by segment",
        ])

    # Margin / profitability queries
    if any(kw in query_lower for kw in ["margin", "profitability", "operating margin", "gross margin"]):
        enhanced_parts.extend([
            "gross margin operating margin net margin",
            "cost of sales revenue",
        ])

    # Growth / decline / change queries
    if any(kw in query_lower for kw in ["growth", "decline", "increase", "decrease", "change", "drove", "driver"]):
        enhanced_parts.extend([
            "growth decline increase decrease driven",
            "fiscal year comparison year over year",
        ])

    return " ".join(enhanced_parts)


# =============================================================================
# V3 Multi-Stage Retrieval: Query Rewrite
# =============================================================================

# Financial section keywords that help expand short queries
_FINANCIAL_SECTION_TERMS = [
    "10-K", "annual report", "financial statement",
    "item 7", "item 8", "management discussion",
    "balance sheet", "income statement", "cash flow",
    "consolidated financial statements",
]


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
    from app.services.llm_service import get_embedding

    original = query.strip()
    query_lower = original.lower()
    tokens = _tokenize_text(original)

    added_terms: list[str] = []

    # Determine intent if not provided
    if intent is None:
        intent = classify_query_intent(original)

    # Strip question words and auxiliaries for expansion detection
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

    # Token count of stripped query
    stripped_tokens = _tokenize_text(stripped)
    is_short = len(stripped_tokens) <= 3

    # Entity extraction: preserve company name / financial term
    entity_terms = [t for t in stripped_tokens if len(t) >= 3 and not t.isdigit()]

    rewritten_parts = [original]

    if intent == QUERY_INTENT_NUMERIC:
        # Numeric queries: add financial statement + ratio/cash/tax context
        if is_short:
            rewritten_parts.extend(_FINANCIAL_SECTION_TERMS[:5])
            added_terms.extend(_FINANCIAL_SECTION_TERMS[:5])
        # Add financial metric terms
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
        # List queries: add business/product context
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
        # Descriptive queries: add explanation/analysis context
        if is_short:
            desc_terms = [
                "management discussion", "analysis", "item 7",
                "overview", "results of operations",
            ]
            for t in desc_terms:
                if t not in query_lower:
                    rewritten_parts.append(t)
                    added_terms.append(t)

    # Always try to add the most relevant financial section terms
    # for very short queries (entity only, e.g. "AMD quick ratio")
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


# =============================================================================
# V3 Multi-Stage Retrieval: Contextual Header
# =============================================================================

# Well-known financial section patterns for header extraction
_FINANCIAL_SECTION_PATTERNS = [
    (r"item\s*7[abc]?\s*[-–—]?\s*.*management", "Item 7 - Management Discussion"),
    (r"item\s*8[abc]?\s*[-–—]?\s*.*financial", "Item 8 - Financial Statements"),
    (r"item\s*1[abc]?\s*[-–—]?\s*.*business", "Item 1 - Business"),
    (r"item\s*1[abc]?\s*[-–—]?\s*.*risk", "Item 1A - Risk Factors"),
    (r"balance\s*sheet", "Balance Sheet"),
    (r"income\s*statement|statement\s*of\s*operations", "Income Statement"),
    (r"cash\s*flow", "Cash Flow Statement"),
    (r"notes?\s*to\s*(consolidated\s*)?financial", "Notes to Financial Statements"),
    (r"consolidated\s*balance", "Consolidated Balance Sheet"),
    (r"selected\s*financial\s*data", "Selected Financial Data"),
    (r"management.*discussion.*analysis", "MD&A"),
    (r"financial.*summary|summary.*financial", "Financial Summary"),
]


def build_contextual_header(
    doc_title: str,
    page_start: int | None,
    section_title: str,
    section_path: str,
    chunk_type: str | None = None,
) -> str:
    """
    Build a contextual header string for a chunk.

    Format:
        "{doc_title} | Page {page} | Section: {section_path} > {section_title} | {chunk_type}"

    This header is prepended to the chunk text when constructing
    the contextual embedding, helping disambiguate sections that
    discuss similar topics in different parts of the document.

    Example:
        "AMD_2022_10K | Page 55 | Section: Item 8 > Consolidated Balance Sheets | table"
    """
    parts = []

    if doc_title:
        parts.append(str(doc_title))
    if page_start is not None:
        parts.append(f"Page {page_start}")
    if section_path or section_title:
        section_str = " > ".join(filter(None, [section_path, section_title]))
        if section_str:
            parts.append(f"Section: {section_str}")
    if chunk_type:
        parts.append(f"Type: {chunk_type}")

    return " | ".join(parts)


def build_contextual_text(cand: dict[str, Any]) -> str:
    """
    Build the full contextual text for a chunk candidate.
    Used when generating contextual embeddings for retrieval.
    """
    header = build_contextual_header(
        doc_title=cand.get("title", ""),
        page_start=cand.get("page_start"),
        section_title=cand.get("section_title", ""),
        section_path=cand.get("section_path", ""),
        chunk_type=cand.get("chunk_type"),
    )
    chunk_text = cand.get("search_text", "") or cand.get("chunk_text", "")
    return f"{header}\n{chunk_text}"


# =============================================================================
# V3 Multi-Stage Retrieval: Section Construction
# =============================================================================

# Page gap threshold for section grouping: consecutive pages within this window
# are considered part of the same section
_SECTION_PAGE_GAP_THRESHOLD = int(_cfg("SECTION_PAGE_GAP_THRESHOLD", 3))


def _group_chunks_into_sections(
    candidates: list[dict[str, Any]],
    page_window: int | None = None,
) -> list[dict[str, Any]]:
    """
    Group a list of chunk candidates into sections.

    A section is defined as:
      - Same document_id
      - Consecutive/nearby pages (within page_window)
      - Same section_path (optional grouping key)

    Each section is returned as a dict with:
      - section_id: unique string
      - doc_id: int
      - page_start, page_end: int
      - section_title: str
      - section_path: str
      - combined_text: str  (concatenated chunk texts)
      - chunk_ids: list[int]
      - chunk_count: int
      - avg_score: float  (average final_score across chunks)

    Returns sections sorted by avg_score descending.
    """
    if page_window is None:
        page_window = _SECTION_PAGE_GAP_THRESHOLD

    if not candidates:
        return []

    # Group by doc_id
    by_doc: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        did = cand.get("document_id")
        if did is not None:
            try:
                by_doc[int(did)].append(cand)
            except (TypeError, ValueError):
                pass

    all_sections: list[dict[str, Any]] = []

    for doc_id, doc_cands in by_doc.items():
        # Sort by page_start
        doc_cands.sort(key=lambda x: to_float(x.get("page_start")) or 0)

        # Group into sections by page proximity
        current_group: list[dict[str, Any]] = []
        group_start_page = None

        def flush_group(group: list[dict[str, Any]], start_page: int | None) -> None:
            if not group:
                return
            # Determine section title (most common in group)
            titles = Counter(c.get("section_title", "") for c in group if c.get("section_title"))
            top_title = titles.most_common(1)[0][0] if titles else ""

            paths = Counter(c.get("section_path", "") for c in group if c.get("section_path"))
            top_path = paths.most_common(1)[0][0] if paths else ""

            pages = [c.get("page_start") for c in group if c.get("page_start") is not None]
            chunk_ids = [c.get("chunk_id") for c in group if c.get("chunk_id") is not None]
            scores = [to_float(c.get("final_score", 0.0)) for c in group]

            combined_text = "\n".join(
                c.get("search_text", "") or c.get("chunk_text", "") or ""
                for c in group
            )

            section = {
                "section_id": f"doc_{doc_id}_page_{start_page if start_page is not None else 0}",
                "doc_id": doc_id,
                "page_start": min(pages) if pages else 0,
                "page_end": max(pages) if pages else 0,
                "section_title": top_title,
                "section_path": top_path,
                "title": group[0].get("title", "") if group else "",
                "combined_text": combined_text,
                "chunk_ids": chunk_ids,
                "chunk_count": len(group),
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "gold_chunks": group,  # preserve for downstream
            }
            all_sections.append(section)

        for cand in doc_cands:
            p = cand.get("page_start")
            if p is None:
                continue
            try:
                page = int(p)
            except (TypeError, ValueError):
                continue

            if current_group and group_start_page is not None:
                if page - group_start_page <= page_window:
                    current_group.append(cand)
                else:
                    # Gap too large: flush and start new group
                    flush_group(current_group, group_start_page)
                    current_group = [cand]
                    group_start_page = page
            else:
                current_group.append(cand)
                group_start_page = page

        flush_group(current_group, group_start_page)

    # Sort sections by avg_score descending
    all_sections.sort(key=lambda x: x.get("avg_score", 0.0), reverse=True)
    return all_sections


def _score_sections_by_lexical(
    sections: list[dict[str, Any]],
    query_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Score sections by lexical match (keyword/BM25 overlap).
    Updates section['lexical_score'] in place.
    """
    terms = query_info.get("important_terms") or query_info.get("unique_terms") or []
    for section in sections:
        text = section.get("combined_text", "").lower()
        section_title = section.get("section_title", "").lower()
        section_path = section.get("section_path", "").lower()
        combined = (section_title + " " + section_path + " " + text).lower()

        # Count term matches
        hits = 0
        for term in terms:
            term_lower = term.lower()
            hits += combined.count(term_lower)

        # Score based on hit density
        coverage = hits / max(1, len(terms))
        lexical_score = min(1.0, coverage + hits * 0.02)
        section["lexical_score"] = lexical_score

    return sections


# =============================================================================
# V3 Multi-Stage Retrieval: Hybrid Score Normalization
# =============================================================================

def _hybrid_normalize(sections: list[dict[str, Any]], score_key: str) -> None:
    """Min-max normalize a score field across sections."""
    vals = [to_float(s.get(score_key, 0.0)) for s in sections]
    max_val = max(vals) if vals else 1.0
    if max_val <= 0:
        max_val = 1.0
    for s in sections:
        s[score_key] = to_float(s.get(score_key, 0.0)) / max_val


# =============================================================================
# V3 Multi-Stage Retrieval: Section-level Retrieval
# =============================================================================
# Bounded LRU Cache for section embeddings
# =============================================================================

from collections import OrderedDict
from threading import Lock

_SECTION_EMBEDDING_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_SECTION_EMBEDDING_CACHE_LOCK = Lock()
_SECTION_EMBEDDING_CACHE_MAX_SIZE = 512


def _get_section_embedding(section: dict[str, Any]) -> list[float]:
    """
    Get or compute embedding for a section's combined text.
    Uses contextual header text for the section.
    Uses a bounded LRU cache to avoid unbounded memory growth.
    """
    section_id = section.get("section_id", "")
    with _SECTION_EMBEDDING_CACHE_LOCK:
        if section_id in _SECTION_EMBEDDING_CACHE:
            _SECTION_EMBEDDING_CACHE.move_to_end(section_id)
            return _SECTION_EMBEDDING_CACHE[section_id]

    # Build contextual header for the section
    header = build_contextual_header(
        doc_title=section.get("title", ""),
        page_start=section.get("page_start"),
        section_title=section.get("section_title", ""),
        section_path=section.get("section_path", ""),
    )
    # Use first 1500 chars of combined text (avoid huge embedding calls)
    text = section.get("combined_text", "")[:1500]
    full_text = f"{header}\n{text}"

    from app.services.llm_service import get_embedding
    emb = get_embedding(full_text)

    with _SECTION_EMBEDDING_CACHE_LOCK:
        if section_id in _SECTION_EMBEDDING_CACHE:
            _SECTION_EMBEDDING_CACHE.move_to_end(section_id)
        else:
            if len(_SECTION_EMBEDDING_CACHE) >= _SECTION_EMBEDDING_CACHE_MAX_SIZE:
                _SECTION_EMBEDDING_CACHE.popitem(last=False)
            _SECTION_EMBEDDING_CACHE[section_id] = emb

    return emb


def _retrieve_sections_from_candidates(
    candidates: list[dict[str, Any]],
    query_info: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Given a set of chunk candidates (from initial retrieval),
    group them into sections and score each section by:
      1. Average embedding similarity (from initial chunk scores)
      2. Lexical keyword overlap
      3. Section-title relevance

    Returns top-k sections sorted by combined score.
    """
    if not candidates:
        return []

    # Step 1: Build sections from candidates
    sections = _group_chunks_into_sections(candidates, page_window=RETRIEVAL_SECTION_PAGE_WINDOW)

    if not sections:
        return []

    # Step 2: Compute REAL embedding similarity per section
    # Get query embedding (for similarity computation)
    query_text = query_info.get("normalized_query", "")
    from app.services.llm_service import get_embedding
    query_emb = get_embedding(query_text) if query_text else []
    query_emb_arr = query_emb if isinstance(query_emb, list) and query_emb else None

    section_emb_scores: dict[str, float] = {}
    for section in sections:
        section_emb = _get_section_embedding(section)
        score = 0.0
        if section_emb and query_emb_arr and len(section_emb) == len(query_emb_arr):
            score = cosine_similarity(section_emb, query_emb_arr)
        section_emb_scores[section["section_id"]] = score
        section["embedding_score"] = score

    # Fallback: if all embeddings are 0, use max chunk score proxy
    if not any(section_emb_scores.values()):
        section_chunk_scores: dict[str, float] = {}
        for cand in candidates:
            section_key = f"doc_{cand.get('document_id')}_page_{cand.get('page_start')}"
            score = to_float(cand.get("final_score", 0.0)) or to_float(cand.get("embedding_score", 0.0))
            if section_key not in section_chunk_scores or score > section_chunk_scores[section_key]:
                section_chunk_scores[section_key] = score
        for section in sections:
            section["embedding_score"] = section_chunk_scores.get(section["section_id"], 0.0)

    # Step 3: Lexical score
    _score_sections_by_lexical(sections, query_info)

    # Step 4: Normalize and combine
    _hybrid_normalize(sections, "embedding_score")
    _hybrid_normalize(sections, "lexical_score")

    # Section title boost: sections with financial section keywords
    for section in sections:
        title_path = (
            section.get("section_title", "").lower() + " " +
            section.get("section_path", "").lower()
        )
        title_hits = sum(
            1 for kw in [
                "balance sheet", "income statement", "cash flow",
                "management discussion", "overview", "business",
                "risk factor", "notes to", "financial summary",
            ]
            if kw in title_path
        )
        section["title_boost"] = min(0.1, title_hits * 0.03)

    # Step 5: Combined score
    for section in sections:
        section["section_score"] = (
            RETRIEVAL_HYBRID_W_DENSE * section.get("embedding_score", 0.0)
            + RETRIEVAL_HYBRID_W_LEXICAL * section.get("lexical_score", 0.0)
            + section.get("title_boost", 0.0)
        )

    sections.sort(key=lambda x: x.get("section_score", 0.0), reverse=True)
    return sections[:top_k]


# =============================================================================
# V3 Multi-Stage Retrieval: Chunk Retrieval within Sections
# =============================================================================

def _retrieve_chunks_within_sections(
    sections: list[dict[str, Any]],
    query_info: dict[str, Any],
    chunks_per_section: int = 5,
) -> list[dict[str, Any]]:
    """
    For each top section, retrieve the best chunks from within that section.
    Uses the original chunk candidates preserved in section['gold_chunks'].

    Returns a flat list of selected chunks (up to sum(chunks_per_section) per section).
    """
    selected_chunks: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for section in sections:
        section_chunks = section.get("gold_chunks", [])
        if not section_chunks:
            continue

        # Sort by final_score
        sorted_chunks = sorted(
            section_chunks,
            key=lambda x: to_float(x.get("final_score", 0.0)),
            reverse=True,
        )

        section_selected = 0
        for chunk in sorted_chunks:
            cid = chunk.get("chunk_id")
            if cid is None or cid in seen_ids:
                continue

            # Add section relevance signal
            chunk["section_score"] = section.get("section_score", 0.0)
            chunk["section_page_start"] = section.get("page_start")
            chunk["section_page_end"] = section.get("page_end")
            chunk["section_title"] = section.get("section_title", "")
            chunk["_from_section_retrieval"] = True

            selected_chunks.append(chunk)
            seen_ids.add(cid)
            section_selected += 1

            if section_selected >= chunks_per_section:
                break

    return selected_chunks


# =============================================================================
# V3 Multi-Stage Retrieval: Enhanced Rerank with Section Relevance
# =============================================================================

def _rerank_with_section_relevance(
    candidates: list[dict[str, Any]],
    query_info: dict[str, Any],
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Re-score chunks by combining their existing scores with section relevance.
    Sections that are highly ranked contribute a bonus to their constituent chunks.

    This is the final pass before deduplication.
    """
    if not sections:
        return candidates

    # Build section score lookup
    section_scores: dict[str, float] = {
        s["section_id"]: s.get("section_score", 0.0)
        for s in sections
    }

    for cand in candidates:
        # Determine which section this chunk belongs to
        doc_id = cand.get("document_id")
        page = cand.get("page_start")
        if doc_id is not None and page is not None:
            section_key = f"doc_{doc_id}_page_{page}"
            sec_score = section_scores.get(section_key, 0.0)
            if sec_score > 0:
                # Boost by section relevance
                section_bonus = RETRIEVAL_SECTION_RELEVANCE_BOOST * sec_score
                cand["final_score"] = to_float(cand.get("final_score", 0.0)) + section_bonus
                cand["_section_relevance_bonus"] = round(section_bonus, 6)

    candidates.sort(key=lambda x: to_float(x.get("final_score", 0.0)), reverse=True)
    return candidates


# =============================================================================
# V3 Multi-Stage Retrieval: Full Multi-stage Pipeline
# =============================================================================

def retrieve_chunks_multistage(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    enhanced_query: str | None = None,
) -> list[dict[str, Any]]:
    """
    V3 Multi-stage retrieval pipeline:

      1. Query understanding (classify + rewrite)
      2. Initial candidate retrieval (vector + lexical + keyword)
      3. Section-level retrieval & scoring
      4. Chunk retrieval within top sections
      5. Multi-signal rerank with section relevance
      6. Deduplication + page clustering

    Returns the same chunk dict format as retrieve_chunks(), with additional
    v3-specific fields (section_score, _from_section_retrieval, etc.).
    """
    # Step 1: Query understanding
    rewrite_result = rewrite_query(query)
    retrieval_query = enhanced_query if enhanced_query else rewrite_result["rewritten_query"]
    intent = rewrite_result["intent"]
    query_info = _normalize_query(retrieval_query)
    if not query_info.get("normalized_query"):
        return []

    candidate_top_k = max(top_k, 20)

    # Step 2: Initial candidate retrieval (same as original)
    lexical_hits = _lexical_recall_from_db(
        query_info=query_info,
        top_k=max(candidate_top_k, HYBRID_HYDRATE_TOP_K),
    )
    vector_hits = _vector_recall_from_qdrant(
        query_info=query_info,
        top_k=max(candidate_top_k, HYBRID_VECTOR_TOP_K),
    )
    if not lexical_hits and not vector_hits:
        return []

    seed_candidates = _merge_recall_candidates(lexical_hits, vector_hits)

    keyword_hits = _keyword_recall_from_candidates(
        query_info=query_info,
        candidates=seed_candidates,
        top_k=max(candidate_top_k, HYBRID_KEYWORD_TOP_K),
    )
    merged = _merge_recall_candidates(seed_candidates, keyword_hits)

    # Secondary financial recall
    secondary_hits = _secondary_financial_recall(
        query_info=query_info,
        top_k=max(candidate_top_k, 60),
    )
    if secondary_hits:
        merged = _merge_recall_candidates(merged, secondary_hits)

    # Initial rerank to get candidate scores
    reranked_initial = _rerank_hybrid_candidates(
        merged,
        query_info=query_info,
        enable_page_diversity=True,
        enable_page_clustering=False,  # Do page clustering at the end
    )

    # Step 3: Section-level retrieval
    # Use top section candidates for section scoring
    section_candidates_for_sections = _group_chunks_into_sections(
        reranked_initial[:RETRIEVAL_CANDIDATE_SECTIONS * 3],
        page_window=RETRIEVAL_SECTION_PAGE_WINDOW,
    )

    # Score these sections
    scored_sections = _retrieve_sections_from_candidates(
        reranked_initial[:candidate_top_k],
        query_info=query_info,
        top_k=RETRIEVAL_CANDIDATE_SECTIONS,
    )

    # Step 4: Chunk retrieval within top sections
    # Get all original candidates that belong to top sections
    section_doc_pages = set()
    for sec in scored_sections:
        section_doc_pages.add((sec["doc_id"], sec["page_start"]))

    # Filter candidates to those in top sections
    section_aware_candidates: list[dict[str, Any]] = []
    for cand in reranked_initial:
        key = (cand.get("document_id"), cand.get("page_start"))
        if key in section_doc_pages:
            section_aware_candidates.append(cand)

    # Also include top candidates outside sections (for diversity)
    for cand in reranked_initial[:5]:
        if cand not in section_aware_candidates:
            section_aware_candidates.append(cand)

    # Re-rank with section relevance
    reranked = _rerank_with_section_relevance(
        section_aware_candidates,
        query_info=query_info,
        sections=scored_sections,
    )

    primary = _deduplicate_candidates(reranked, top_k=candidate_top_k)
    expanded = _expand_neighbor_chunks(
        primary,
        target_limit=max(candidate_top_k, len(primary) + 2),
    )
    expanded = _cap_page_duplicates(expanded, top_k=candidate_top_k)
    expanded.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    # Page clustering as final pass
    page_cluster_scores = _smooth_page_scores(expanded, alpha=RETRIEVAL_PAGE_CLUSTER_ALPHA)
    if page_cluster_scores:
        for cand in expanded:
            p = cand.get("page_start")
            if p is not None:
                try:
                    cluster_bonus = page_cluster_scores.get(int(p), 0.0) - to_float(cand.get("final_score", 0.0))
                    if cluster_bonus > 0.01:
                        cand["final_score"] += cluster_bonus
                        cand["_page_cluster_bonus"] = round(cluster_bonus, 6)
                except (TypeError, ValueError):
                    pass
        expanded.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    # V4: Apply LLM reranker before final truncation
    if V4_ENABLE_LLM_RERANK and len(expanded) > 1:
        try:
            reranked = rerank_with_llm(
                query=query,
                intent=intent,
                candidates=expanded,
                top_n=None,
            )
            # If LLM reranking succeeded, use the combined score for sorting
            if reranked and any(c.get("llm_rerank_applied", False) for c in reranked):
                expanded = reranked
        except Exception:
            pass

    final_hits = expanded[:top_k]

    # Build results
    results: list[dict[str, Any]] = []
    for item in final_hits:
        results.append({
            "chunk_id": item.get("chunk_id"),
            "document_id": item.get("document_id"),
            "chunk_index": item.get("chunk_index"),
            "score": round(to_float(item.get("final_score")), 6),
            "rerank_score": round(to_float(item.get("final_score")), 6),
            "embedding_score": round(to_float(item.get("embedding_score")), 6),
            "keyword_score": round(to_float(item.get("keyword_score")), 6),
            "bm25_score": round(to_float(item.get("bm25_score")), 6),
            "title_match_score": round(to_float(item.get("title_match_score")), 6),
            "section_match_score": round(to_float(item.get("section_match_score")), 6),
            "coverage_score": round(to_float(item.get("coverage_score")), 6),
            "matched_term_count": int(to_float(item.get("matched_term_count"))),
            "title": item.get("title", ""),
            "section_path": item.get("section_path", ""),
            "section_title": item.get("section_title", ""),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "chunk_type": item.get("chunk_type"),
            "chunk_text": item.get("chunk_text", ""),
            "search_text": item.get("search_text", ""),
            "lexical_text": item.get("lexical_text", ""),
            "term_hits": item.get("term_hits", {}),
            "term_hit_detail": item.get("term_hit_detail", {}),
            "is_neighbor": bool(item.get("is_neighbor", False)),
            # V3 new fields
            "_retrieval_query": retrieval_query if retrieval_query != query else None,
            "_numeric_boost": round(to_float(item.get("_numeric_boost", 0.0)), 6),
            "_table_boost": round(to_float(item.get("_table_boost", 0.0)), 6),
            "_query_aware_boost": round(to_float(item.get("_query_aware_boost", 0.0)), 6),
            "_anti_noise_penalty": round(to_float(item.get("_anti_noise_penalty", 0.0)), 6),
            "_page_cluster_bonus": round(to_float(item.get("_page_cluster_bonus", 0.0)), 6),
            "_section_relevance_bonus": round(to_float(item.get("_section_relevance_bonus", 0.0)), 6),
            "_query_intent": item.get("_query_intent", intent),
            "_from_section_retrieval": bool(item.get("_from_section_retrieval", False)),
            "_section_score": round(to_float(item.get("section_score", 0.0)), 6),
            "_rewrite_intent": intent,
            # V4 LLM reranker fields
            "llm_relevance_score": round(to_float(item.get("llm_relevance_score", 0.0)), 6),
            "llm_rationale": str(item.get("llm_rationale", "")),
            "llm_combined_score": round(to_float(item.get("llm_combined_score", item.get("final_score", 0.0))), 6),
            "llm_rerank_applied": bool(item.get("llm_rerank_applied", False)),
        })

    return results


def retrieve_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    enhanced_query: str | None = None,
    use_multistage: bool = False,
) -> list[dict[str, Any]]:
    """
    Two-stage retrieval: fetch large candidate pool → rerank → return top_k.

    Parameters
    ----------
    query : str
        Original user query (used as-is for answer generation).
    top_k : int
        Final number of chunks to return after reranking.
    enhanced_query : str, optional
        If provided, this query is used for retrieval while `query` is used as-is.
        Allows query augmentation without changing the answer question.
    use_multistage : bool, optional
        If True, uses the V3 multi-stage retrieval pipeline (doc → section → chunk).
        Default: False (uses the standard two-stage retrieval).
    """
    if use_multistage or RETRIEVAL_USE_MULTISTAGE:
        return retrieve_chunks_multistage(
            query=query,
            top_k=top_k,
            enhanced_query=enhanced_query,
        )
    # Use enhanced query for retrieval if provided, otherwise use original
    retrieval_query = enhanced_query if enhanced_query else query

    query_info = _normalize_query(retrieval_query)
    if not query_info.get("normalized_query"):
        return []

    # Stage 1: Retrieve large candidate pool (at least 20, default 20 here
    # but actual internal fetch is controlled by HYBRID_*_TOP_K constants)
    candidate_top_k = max(top_k, 20)

    lexical_hits = _lexical_recall_from_db(
        query_info=query_info,
        top_k=max(candidate_top_k, HYBRID_HYDRATE_TOP_K),
    )

    vector_hits = _vector_recall_from_qdrant(
        query_info=query_info,
        top_k=max(candidate_top_k, HYBRID_VECTOR_TOP_K),
    )

    if not lexical_hits and not vector_hits:
        return []

    seed_candidates = _merge_recall_candidates(lexical_hits, vector_hits)

    keyword_hits = _keyword_recall_from_candidates(
        query_info=query_info,
        candidates=seed_candidates,
        top_k=max(candidate_top_k, HYBRID_KEYWORD_TOP_K),
    )

    merged = _merge_recall_candidates(seed_candidates, keyword_hits)

    # Secondary financial recall: activated when query mentions ratio/liquidity/etc.
    # This helps surface balance-sheet pages (current assets, liabilities, cash) that
    # the primary semantic search may miss because the vocabulary doesn't align.
    secondary_hits = _secondary_financial_recall(
        query_info=query_info,
        top_k=max(candidate_top_k, 60),
    )
    if secondary_hits:
        merged = _merge_recall_candidates(merged, secondary_hits)

    # Stage 2: Rerank the full candidate pool, then truncate to top_k
    reranked = _rerank_hybrid_candidates(merged, query_info=query_info)
    primary = _deduplicate_candidates(reranked, top_k=candidate_top_k)
    expanded = _expand_neighbor_chunks(
        primary,
        target_limit=max(candidate_top_k, len(primary) + 2),
    )
    # Enforce page-diversity cap after neighbor expansion to ensure
    # we don't flood the top-k with chunks from the same few pages
    expanded = _cap_page_duplicates(expanded, top_k=candidate_top_k)

    expanded.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    # Final truncation to requested top_k AFTER all reranking/dedup/diversity
    final_hits = expanded[:top_k]

    results: list[dict[str, Any]] = []
    for item in final_hits:
        results.append(
            {
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "chunk_index": item.get("chunk_index"),
                "score": round(to_float(item.get("final_score")), 6),
                "rerank_score": round(to_float(item.get("final_score")), 6),
                "embedding_score": round(to_float(item.get("embedding_score")), 6),
                "keyword_score": round(to_float(item.get("keyword_score")), 6),
                "bm25_score": round(to_float(item.get("bm25_score")), 6),
                "title_match_score": round(to_float(item.get("title_match_score")), 6),
                "section_match_score": round(to_float(item.get("section_match_score")), 6),
                "coverage_score": round(to_float(item.get("coverage_score")), 6),
                "matched_term_count": int(to_float(item.get("matched_term_count"))),
                "title": item.get("title", ""),
                "section_path": item.get("section_path", ""),
                "section_title": item.get("section_title", ""),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "chunk_type": item.get("chunk_type"),
                "chunk_text": item.get("chunk_text", ""),
                "search_text": item.get("search_text", ""),
                "lexical_text": item.get("lexical_text", ""),
                "term_hits": item.get("term_hits", {}),
                "term_hit_detail": item.get("term_hit_detail", {}),
                "is_neighbor": bool(item.get("is_neighbor", False)),
                # Debug: enhanced query used for retrieval (if any)
                "_retrieval_query": retrieval_query if retrieval_query != query else None,
                # New P0 rerank signal breakdowns
                "_numeric_boost": round(to_float(item.get("_numeric_boost", 0.0)), 6),
                "_table_boost": round(to_float(item.get("_table_boost", 0.0)), 6),
                "_query_aware_boost": round(to_float(item.get("_query_aware_boost", 0.0)), 6),
                "_anti_noise_penalty": round(to_float(item.get("_anti_noise_penalty", 0.0)), 6),
                "_page_cluster_bonus": round(to_float(item.get("_page_cluster_bonus", 0.0)), 6),
                "_query_intent": item.get("_query_intent", "unknown"),
            }
        )

    return results