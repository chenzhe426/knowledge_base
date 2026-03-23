from __future__ import annotations

from typing import Any

from app.services.vector_store import vector_store


def upsert_chunk_embedding(
    *,
    chunk_id: int | None,
    embedding: list[float] | None,
    payload: dict[str, Any],
) -> None:
    if not chunk_id or not embedding:
        return

    try:
        vector_store.upsert_chunks(
            [
                {
                    "chunk_id": int(chunk_id),
                    "document_id": payload.get("document_id"),
                    "chunk_index": payload.get("chunk_index"),
                    "embedding": embedding,
                    "doc_title": payload.get("doc_title", ""),
                    "section_path": payload.get("section_path", ""),
                    "section_title": payload.get("section_title", ""),
                    "chunk_type": payload.get("chunk_type"),
                    "page_start": payload.get("page_start"),
                    "page_end": payload.get("page_end"),
                }
            ]
        )
    except Exception:
        import logging
        logging.exception("qdrant upsert failed, chunk_id=%s", chunk_id)


def delete_document_embeddings(document_id: int) -> None:
    if not document_id:
        return

    try:
        vector_store.delete_document_chunks(int(document_id))
    except Exception:
        import logging
        logging.exception("qdrant delete failed, document_id=%s", document_id)