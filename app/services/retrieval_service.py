import math
import re
from collections import Counter, defaultdict
from typing import Any
from app.services.vector_store import vector_store

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

QUERY_MIN_TERM_LEN = int(_cfg("QUERY_MIN_TERM_LEN", 2))
QUERY_STOPWORDS = set(_cfg("QUERY_STOPWORDS", []))


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

    # 中文 -> 英文/数字 边界
    text = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9]+)", r"\1 \2", text)
    # 英文/数字 -> 中文 边界
    text = re.sub(r"([A-Za-z0-9]+)([\u4e00-\u9fff])", r"\1 \2", text)
    # 压缩空白
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
    metadata = safe_json_loads(safe_get(row, "metadata_json"), default={}) or {}
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
        "doc_title": normalize_whitespace(
            safe_get(row, "doc_title")
            or safe_get(row, "title")
            or metadata.get("doc_title", "")
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
    doc_title = normalize_whitespace(cand.get("doc_title", ""))
    section_title = normalize_whitespace(cand.get("section_title", ""))
    section_path = normalize_whitespace(cand.get("section_path", ""))
    search_text = normalize_whitespace(cand.get("search_text", ""))
    lexical_text = normalize_whitespace(cand.get("lexical_text", ""))

    return {
        "doc_title": doc_title,
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

        title_hit = _term_occurrence_detail(term, texts["doc_title"])
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

def _vector_recall_from_candidates(
    query_info: dict[str, Any],
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    query_text = query_info.get("normalized_query", "")
    return vector_store.score_candidates(query_text, candidates, top_k)

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


def _rerank_hybrid_candidates(candidates: list[dict[str, Any]], query_info: dict[str, Any]) -> list[dict[str, Any]]:
    reranked: list[dict[str, Any]] = []
    query_terms = query_info.get("important_terms") or query_info.get("unique_terms") or []

    for cand in candidates:
        text = cand.get("search_text", "") or cand.get("chunk_text", "")

        embedding_score = to_float(cand.get("embedding_score"))
        keyword_score = to_float(cand.get("keyword_score"))
        bm25_score = to_float(cand.get("bm25_score"))
        title_score = to_float(cand.get("title_match_score"))
        section_score = to_float(cand.get("section_match_score"))
        coverage_score = to_float(cand.get("coverage_score"))

        coverage_bonus = 0.08 * coverage_score if query_terms else 0.0

        if len(query_terms) <= 2:
            embedding_weight = max(0.0, RETRIEVAL_WEIGHT_EMBEDDING - 0.05)
            keyword_weight = RETRIEVAL_WEIGHT_KEYWORD + 0.03
            bm25_weight = RETRIEVAL_WEIGHT_BM25 + 0.02
        else:
            embedding_weight = RETRIEVAL_WEIGHT_EMBEDDING
            keyword_weight = RETRIEVAL_WEIGHT_KEYWORD
            bm25_weight = RETRIEVAL_WEIGHT_BM25

        final_score = (
            embedding_weight * embedding_score
            + keyword_weight * keyword_score
            + bm25_weight * bm25_score
            + RETRIEVAL_WEIGHT_TITLE * title_score
            + RETRIEVAL_WEIGHT_SECTION * section_score
            + coverage_bonus
            + _metadata_bonus(cand)
            - _length_penalty(text)
        )

        item = dict(cand)
        item["final_score"] = final_score
        reranked.append(item)

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

                    neighbor_cand["final_score"] = center_score * (0.88 - (offset - 1) * 0.06)
                    neighbor_cand["is_neighbor"] = True

                    results.append(neighbor_cand)
                    seen_ids.add(neighbor_id)

                    if len(results) >= target_limit:
                        return results[:target_limit]

    return results[:target_limit]


def retrieve_chunks(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    query_info = _normalize_query(query)
    if not query_info.get("normalized_query"):
        return []

    lexical_hits = _lexical_recall_from_db(
        query_info=query_info,
        top_k=max(top_k, HYBRID_HYDRATE_TOP_K),
    )
    if not lexical_hits:
        return []

    vector_hits = _vector_recall_from_candidates(
        query_info=query_info,
        candidates=lexical_hits,
        top_k=max(top_k, HYBRID_VECTOR_TOP_K),
    )
    keyword_hits = _keyword_recall_from_candidates(
        query_info=query_info,
        candidates=lexical_hits,
        top_k=max(top_k, HYBRID_KEYWORD_TOP_K),
    )

    merged = _merge_recall_candidates(lexical_hits, vector_hits, keyword_hits)
    reranked = _rerank_hybrid_candidates(merged, query_info=query_info)
    primary = _deduplicate_candidates(reranked, top_k=top_k)
    expanded = _expand_neighbor_chunks(
        primary,
        target_limit=max(top_k, len(primary) + 2),
    )

    expanded.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    final_hits = expanded[:top_k]

    results: list[dict[str, Any]] = []
    for item in final_hits:
        results.append(
            {
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "chunk_index": item.get("chunk_index"),
                "score": round(to_float(item.get("final_score")), 6),
                "embedding_score": round(to_float(item.get("embedding_score")), 6),
                "keyword_score": round(to_float(item.get("keyword_score")), 6),
                "bm25_score": round(to_float(item.get("bm25_score")), 6),
                "title_match_score": round(to_float(item.get("title_match_score")), 6),
                "section_match_score": round(to_float(item.get("section_match_score")), 6),
                "coverage_score": round(to_float(item.get("coverage_score")), 6),
                "matched_term_count": int(to_float(item.get("matched_term_count"))),
                "doc_title": item.get("doc_title", ""),
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
            }
        )

    return results