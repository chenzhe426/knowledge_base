import math
import re
from collections import Counter, defaultdict
from typing import Any

from app.config import (
    DEFAULT_TOP_K,
    HYBRID_KEYWORD_TOP_K,
    HYBRID_VECTOR_TOP_K,
    KEYWORD_EXACT_MATCH_WEIGHT,
    KEYWORD_SUBSTRING_MATCH_WEIGHT,
    QUERY_MIN_TERM_LEN,
    QUERY_STOPWORDS,
    RETRIEVAL_DEDUP_SIM_THRESHOLD,
    RETRIEVAL_ENABLE_NEIGHBOR_EXPANSION,
    RETRIEVAL_MAX_SAME_SECTION,
    RETRIEVAL_NEIGHBOR_WINDOW,
    RETRIEVAL_WEIGHT_EMBEDDING,
    RETRIEVAL_WEIGHT_KEYWORD,
    RETRIEVAL_WEIGHT_SECTION,
    RETRIEVAL_WEIGHT_TITLE,
    SECTION_MATCH_WEIGHT,
    TITLE_MATCH_WEIGHT,
)
from app.db import get_all_chunks, get_chunks_by_document_id
from app.services.common import (
    normalize_embedding,
    normalize_whitespace,
    safe_get,
    safe_json_loads,
    to_float,
)
from app.services.llm_service import get_embedding


def _tokenize_query(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    parts = re.findall(r"[A-Za-z][A-Za-z0-9_./-]*|[0-9]+|[\u4e00-\u9fff]+", text)
    tokens: list[str] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_./-]*", part) or re.fullmatch(r"[0-9]+", part):
            tokens.append(part)
            tokens.append(part.lower())
            continue

        tokens.append(part)
        if len(part) >= 4:
            for n in (2, 3, 4):
                for i in range(0, len(part) - n + 1):
                    tokens.append(part[i : i + n])

    return [t for t in tokens if t]


def _normalize_query(query: str) -> dict[str, Any]:
    normalized = normalize_whitespace(query)
    tokens = _tokenize_query(normalized)

    filtered = []
    for t in tokens:
        t_norm = t.strip()
        if not t_norm:
            continue
        if len(t_norm) < QUERY_MIN_TERM_LEN and not re.search(r"[A-Za-z0-9]", t_norm):
            continue
        if t_norm in QUERY_STOPWORDS:
            continue
        filtered.append(t_norm)

    unique_terms = list(dict.fromkeys(filtered))
    important_terms = []

    for t in unique_terms:
        if (
            re.search(r"[A-Z]", t)
            or re.search(r"[_./-]", t)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_./-]*", t)
            or len(t) >= 4
        ):
            important_terms.append(t)

    if not important_terms:
        important_terms = unique_terms[:]

    return {
        "raw_query": query,
        "normalized_query": normalized,
        "terms": filtered,
        "unique_terms": unique_terms,
        "important_terms": important_terms,
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


def _term_occurrence_score(term: str, text: str) -> float:
    if not term or not text:
        return 0.0

    exact_count = text.count(term)
    if exact_count > 0:
        return min(3.0, exact_count * KEYWORD_EXACT_MATCH_WEIGHT)

    lowered_term = term.lower()
    lowered_text = text.lower()
    sub_count = lowered_text.count(lowered_term)
    if sub_count > 0:
        return min(2.0, sub_count * KEYWORD_SUBSTRING_MATCH_WEIGHT)

    return 0.0


def _compute_keyword_components(query_info: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    terms = query_info.get("important_terms") or query_info.get("unique_terms") or []

    chunk_text = normalize_whitespace(safe_get(row, "chunk_text", ""))
    doc_title = normalize_whitespace(
        safe_get(row, "doc_title")
        or safe_get(row, "title")
        or safe_json_loads(safe_get(row, "metadata_json"), {}).get("doc_title", "")
    )
    section_path = normalize_whitespace(safe_get(row, "section_path", ""))
    section_title = normalize_whitespace(
        safe_get(row, "section_title")
        or safe_json_loads(safe_get(row, "metadata_json"), {}).get("section_title", "")
    )

    keyword_raw = 0.0
    title_raw = 0.0
    section_raw = 0.0
    term_hits: Counter[str] = Counter()

    target_section_text = " ".join([section_path, section_title]).strip()

    for term in terms:
        score = _term_occurrence_score(term, chunk_text)
        if score > 0:
            keyword_raw += score
            term_hits[term] += 1

        if term and doc_title:
            if term in doc_title or term.lower() in doc_title.lower():
                title_raw += TITLE_MATCH_WEIGHT

        if term and target_section_text:
            if term in target_section_text or term.lower() in target_section_text.lower():
                section_raw += SECTION_MATCH_WEIGHT

    keyword_score = min(1.0, keyword_raw / 4.0)
    title_match_score = min(1.0, title_raw / 3.0)
    section_match_score = min(1.0, section_raw / 3.0)

    return {
        "keyword_score": keyword_score,
        "title_match_score": title_match_score,
        "section_match_score": section_match_score,
        "term_hits": dict(term_hits),
    }


def _row_to_candidate(row: Any) -> dict[str, Any]:
    metadata = safe_json_loads(safe_get(row, "metadata_json"), default={})
    return {
        "chunk_id": safe_get(row, "id"),
        "document_id": safe_get(row, "document_id"),
        "chunk_index": safe_get(row, "chunk_index"),
        "chunk_text": safe_get(row, "chunk_text", ""),
        "embedding": normalize_embedding(safe_get(row, "embedding")),
        "doc_title": safe_get(row, "doc_title")
        or safe_get(row, "title")
        or metadata.get("doc_title", ""),
        "section_path": safe_get(row, "section_path", ""),
        "section_title": safe_get(row, "section_title") or metadata.get("section_title", ""),
        "page_start": safe_get(row, "page_start"),
        "page_end": safe_get(row, "page_end"),
        "block_start_index": safe_get(row, "block_start_index"),
        "block_end_index": safe_get(row, "block_end_index"),
        "chunk_type": safe_get(row, "chunk_type"),
        "metadata": metadata,
        "embedding_score": 0.0,
        "keyword_score": 0.0,
        "title_match_score": 0.0,
        "section_match_score": 0.0,
        "final_score": 0.0,
        "term_hits": {},
    }


def _vector_recall(query_info: dict[str, Any], rows: list[Any], top_k: int) -> list[dict[str, Any]]:
    query_embedding = get_embedding(query_info.get("normalized_query", ""))
    candidates: list[dict[str, Any]] = []

    for row in rows:
        cand = _row_to_candidate(row)
        emb = cand.get("embedding") or []
        cand["embedding_score"] = cosine_similarity(query_embedding, emb)
        candidates.append(cand)

    candidates.sort(key=lambda x: x.get("embedding_score", 0.0), reverse=True)
    return candidates[:top_k]


def _keyword_recall(query_info: dict[str, Any], rows: list[Any], top_k: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for row in rows:
        cand = _row_to_candidate(row)
        parts = _compute_keyword_components(query_info, cand)
        cand.update(parts)

        score = (
            0.6 * cand["keyword_score"]
            + 0.25 * cand["title_match_score"]
            + 0.15 * cand["section_match_score"]
        )
        cand["final_score"] = score
        if score > 0:
            candidates.append(cand)

    candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return candidates[:top_k]


def _merge_recall_candidates(
    vector_hits: list[dict[str, Any]],
    keyword_hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[Any, dict[str, Any]] = {}

    def merge_one(item: dict[str, Any]):
        chunk_id = item.get("chunk_id")
        if chunk_id not in merged:
            merged[chunk_id] = dict(item)
            return

        current = merged[chunk_id]
        for field in [
            "embedding_score",
            "keyword_score",
            "title_match_score",
            "section_match_score",
            "final_score",
        ]:
            current[field] = max(to_float(current.get(field)), to_float(item.get(field)))

        if not current.get("term_hits") and item.get("term_hits"):
            current["term_hits"] = item["term_hits"]

    for item in vector_hits:
        merge_one(item)
    for item in keyword_hits:
        merge_one(item)

    return list(merged.values())


def _rerank_hybrid_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reranked = []

    for cand in candidates:
        text = cand.get("chunk_text", "") or ""
        text_len = len(text)
        length_penalty = 0.0
        if text_len < 20:
            length_penalty = 0.05
        elif text_len > 1200:
            length_penalty = 0.08

        final_score = (
            RETRIEVAL_WEIGHT_EMBEDDING * to_float(cand.get("embedding_score"))
            + RETRIEVAL_WEIGHT_KEYWORD * to_float(cand.get("keyword_score"))
            + RETRIEVAL_WEIGHT_TITLE * to_float(cand.get("title_match_score"))
            + RETRIEVAL_WEIGHT_SECTION * to_float(cand.get("section_match_score"))
            - length_penalty
        )

        item = dict(cand)
        item["final_score"] = final_score
        reranked.append(item)

    reranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return reranked


def _text_similarity_for_dedup(a: str, b: str) -> float:
    tokens_a = set(_tokenize_query(a.lower()))
    tokens_b = set(_tokenize_query(b.lower()))
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
                cand.get("chunk_text", ""),
                chosen.get("chunk_text", ""),
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
            grouped[doc_id].append(cand)

    for doc_id, doc_cands in grouped.items():
        doc_rows = get_chunks_by_document_id(doc_id) or []
        index_map = {safe_get(r, "chunk_index"): r for r in doc_rows}

        for cand in doc_cands:
            cid = cand.get("chunk_id")
            if cid not in seen_ids:
                results.append(cand)
                seen_ids.add(cid)

            center_index = cand.get("chunk_index")
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
                    neighbor_cand["final_score"] = to_float(cand.get("final_score")) * 0.85
                    results.append(neighbor_cand)
                    seen_ids.add(neighbor_id)

                    if len(results) >= target_limit:
                        return results[:target_limit]

    return results[:target_limit]


def retrieve_chunks(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    rows = get_all_chunks() or []
    if not rows:
        return []

    query_info = _normalize_query(query)

    vector_hits = _vector_recall(
        query_info=query_info,
        rows=rows,
        top_k=max(top_k, HYBRID_VECTOR_TOP_K),
    )
    keyword_hits = _keyword_recall(
        query_info=query_info,
        rows=rows,
        top_k=max(top_k, HYBRID_KEYWORD_TOP_K),
    )

    merged = _merge_recall_candidates(vector_hits, keyword_hits)
    reranked = _rerank_hybrid_candidates(merged)

    primary = _deduplicate_candidates(reranked, top_k=top_k)
    expanded = _expand_neighbor_chunks(primary, target_limit=max(top_k, len(primary) + 2))

    expanded.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    final_hits = expanded[:top_k]

    results = []
    for item in final_hits:
        results.append(
            {
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "chunk_index": item.get("chunk_index"),
                "score": round(to_float(item.get("final_score")), 6),
                "embedding_score": round(to_float(item.get("embedding_score")), 6),
                "keyword_score": round(to_float(item.get("keyword_score")), 6),
                "title_match_score": round(to_float(item.get("title_match_score")), 6),
                "section_match_score": round(to_float(item.get("section_match_score")), 6),
                "doc_title": item.get("doc_title", ""),
                "section_path": item.get("section_path", ""),
                "section_title": item.get("section_title", ""),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "chunk_type": item.get("chunk_type"),
                "chunk_text": item.get("chunk_text", ""),
                "term_hits": item.get("term_hits", {}),
            }
        )

    return results