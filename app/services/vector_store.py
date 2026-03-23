from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.services.llm_service import get_embedding


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_embedding(emb: Any) -> list[float]:
    if not emb:
        return []

    try:
        vector = [float(x) for x in emb]
    except (TypeError, ValueError):
        return []

    if not vector:
        return []

    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        return []

    norm = np.linalg.norm(arr)
    if norm == 0:
        return []

    return (arr / norm).astype(np.float32).tolist()


@lru_cache(maxsize=512)
def _get_query_embedding_cached(query_text: str) -> tuple[float, ...] | None:
    emb = get_embedding(query_text)
    normalized = _normalize_embedding(emb)
    if not normalized:
        return None
    return tuple(normalized)


class BaseVectorStore:
    def score_candidates(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def delete_document_chunks(self, document_id: int) -> None:
        raise NotImplementedError

    def ensure_collection(self) -> None:
        raise NotImplementedError


class QdrantVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.url = _env_str("QDRANT_URL", "http://localhost:6333")
        self.api_key = _env_str("QDRANT_API_KEY", "")
        self.collection_name = _env_str("QDRANT_COLLECTION_NAME", "kb_chunks")
        self.vector_name = _env_str("QDRANT_VECTOR_NAME", "dense")
        self.embedding_dim = _env_int("EMBEDDING_DIM", 1536)
        self.enabled = bool(self.url)

        self.client: QdrantClient | None = None
        if self.enabled:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key or None,
            )

    def ensure_collection(self) -> None:
        if not self.client:
            return

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.vector_name: VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                )
            },
        )

    def score_candidates(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not query_text or not candidates or top_k <= 0:
            return []

        query_embedding = _get_query_embedding_cached(query_text)
        if not query_embedding:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim != 1 or q.size == 0:
            return []

        valid_candidates: list[dict[str, Any]] = []
        vectors: list[np.ndarray] = []

        for cand in candidates:
            emb = cand.get("embedding")
            normalized = _normalize_embedding(emb)
            if not normalized:
                continue

            v = np.asarray(normalized, dtype=np.float32)
            if v.ndim != 1 or v.size != q.size:
                continue

            valid_candidates.append(cand)
            vectors.append(v)

        if not vectors:
            return []

        matrix = np.vstack(vectors)  # [N, D]
        scores = matrix @ q          # [N]

        ranked_idx = np.argsort(-scores)[:top_k]

        results: list[dict[str, Any]] = []
        for idx in ranked_idx:
            score = float(scores[idx])
            if score <= 0:
                continue

            item = dict(valid_candidates[idx])
            item["embedding_score"] = score
            item["final_score"] = max(
                score,
                _to_float(item.get("final_score")),
            )
            results.append(item)

        return results

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.client or not query_text or top_k <= 0:
            return []

        query_embedding = _get_query_embedding_cached(query_text)
        if not query_embedding:
            return []

        self.ensure_collection()

        query_filter = self._build_filter(filters)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(query_embedding),
            using=self.vector_name,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        points = getattr(response, "points", None) or []
        results: list[dict[str, Any]] = []

        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id", point.id)

            try:
                normalized_chunk_id = int(chunk_id)
            except (TypeError, ValueError):
                continue

            item = {
                "chunk_id": normalized_chunk_id,
                "document_id": payload.get("document_id"),
                "chunk_index": payload.get("chunk_index"),
                "embedding_score": _to_float(point.score),
                "final_score": _to_float(point.score),
                "doc_title": payload.get("doc_title", ""),
                "section_path": payload.get("section_path", ""),
                "section_title": payload.get("section_title", ""),
                "chunk_type": payload.get("chunk_type"),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "payload": payload,
            }
            results.append(item)

        return results

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not self.client or not chunks:
            return

        self.ensure_collection()

        points: list[PointStruct] = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", chunk.get("id"))
            if chunk_id is None:
                continue

            try:
                point_id = int(chunk_id)
            except (TypeError, ValueError):
                continue

            embedding = chunk.get("embedding")
            normalized = _normalize_embedding(embedding)
            if not normalized:
                continue

            payload = {
                "chunk_id": point_id,
                "document_id": chunk.get("document_id"),
                "chunk_index": chunk.get("chunk_index"),
                "doc_title": chunk.get("doc_title", ""),
                "section_path": chunk.get("section_path", ""),
                "section_title": chunk.get("section_title", ""),
                "chunk_type": chunk.get("chunk_type"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector={self.vector_name: normalized},
                    payload=payload,
                )
            )

        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def delete_document_chunks(self, document_id: int) -> None:
        if not self.client:
            return

        self.ensure_collection()

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> Filter | None:
        if not filters:
            return None

        must_conditions: list[FieldCondition] = []
        for key, value in filters.items():
            if value is None:
                continue
            must_conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )

        if not must_conditions:
            return None

        return Filter(must=must_conditions)


vector_store = QdrantVectorStore()