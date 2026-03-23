from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from app.services.llm_service import get_embedding


@lru_cache(maxsize=512)
def _get_query_embedding_cached(query_text: str) -> tuple[float, ...] | None:
    emb = get_embedding(query_text)
    if not emb:
        return None
    return tuple(float(x) for x in emb)


class BaseVectorStore:
    def score_candidates(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class NumpyVectorStore(BaseVectorStore):
    def score_candidates(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not query_text or not candidates:
            return []

        query_embedding = _get_query_embedding_cached(query_text)
        if not query_embedding:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim != 1 or q.size == 0:
            return []

        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        valid_candidates: list[dict[str, Any]] = []
        vectors: list[np.ndarray] = []

        for cand in candidates:
            emb = cand.get("embedding") or []
            if not emb:
                continue

            v = np.asarray(emb, dtype=np.float32)
            if v.ndim != 1 or v.size != q.size:
                continue

            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue

            valid_candidates.append(cand)
            vectors.append(v / v_norm)

        if not vectors:
            return []

        matrix = np.vstack(vectors)   # [N, D]
        scores = matrix @ q           # [N]

        ranked_idx = np.argsort(-scores)[:top_k]

        results: list[dict[str, Any]] = []
        for idx in ranked_idx:
            score = float(scores[idx])
            if score <= 0:
                continue
            item = dict(valid_candidates[idx])
            item["embedding_score"] = score
            results.append(item)

        return results


vector_store = NumpyVectorStore()