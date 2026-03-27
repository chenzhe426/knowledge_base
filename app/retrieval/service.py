"""
Retrieval service: public API (retrieve_chunks) and two-stage pipeline.
"""
from __future__ import annotations

from typing import Any

from .config import DEFAULT_TOP_K


def retrieve_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    enhanced_query: str | None = None,
    use_multistage: bool = False,
) -> list[dict[str, Any]]:
    """
    Two-stage retrieval: fetch large candidate pool → rerank → return top_k.
    """
    # Local imports to avoid circular dependency chain:
    # app/retrieval/__init__.py → query_understanding → app.services.common →
    # app.services.__init__ → qa_service → retrieval_service → service → multistage
    from .config import (
        HYBRID_HYDRATE_TOP_K,
        HYBRID_KEYWORD_TOP_K,
        HYBRID_VECTOR_TOP_K,
        RETRIEVAL_USE_MULTISTAGE,
    )
    from app.services.common import to_float
    from .diversity import (
        _cap_page_duplicates,
        _deduplicate_candidates,
        _expand_neighbor_chunks,
    )
    from .query_understanding import _normalize_query
    from .recall import (
        _keyword_recall_from_candidates,
        _lexical_recall_from_db,
        _merge_recall_candidates,
        _secondary_financial_recall,
        _vector_recall_from_qdrant,
    )
    from .rerank import _rerank_hybrid_candidates

    if use_multistage or RETRIEVAL_USE_MULTISTAGE:
        from .multistage import retrieve_chunks_multistage
        return retrieve_chunks_multistage(
            query=query,
            top_k=top_k,
            enhanced_query=enhanced_query,
        )

    retrieval_query = enhanced_query if enhanced_query else query

    query_info = _normalize_query(retrieval_query)
    if not query_info.get("normalized_query"):
        return []

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

    secondary_hits = _secondary_financial_recall(
        query_info=query_info,
        top_k=max(candidate_top_k, 60),
    )
    if secondary_hits:
        merged = _merge_recall_candidates(merged, secondary_hits)

    reranked = _rerank_hybrid_candidates(merged, query_info=query_info)
    primary = _deduplicate_candidates(reranked, top_k=candidate_top_k)
    expanded = _expand_neighbor_chunks(
        primary,
        target_limit=max(candidate_top_k, len(primary) + 2),
    )
    expanded = _cap_page_duplicates(expanded, top_k=candidate_top_k)

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
                "_retrieval_query": retrieval_query if retrieval_query != query else None,
                "_numeric_boost": round(to_float(item.get("_numeric_boost", 0.0)), 6),
                "_table_boost": round(to_float(item.get("_table_boost", 0.0)), 6),
                "_query_aware_boost": round(to_float(item.get("_query_aware_boost", 0.0)), 6),
                "_anti_noise_penalty": round(to_float(item.get("_anti_noise_penalty", 0.0)), 6),
                "_page_cluster_bonus": round(to_float(item.get("_page_cluster_bonus", 0.0)), 6),
                "_query_intent": item.get("_query_intent", "unknown"),
            }
        )

    return results
