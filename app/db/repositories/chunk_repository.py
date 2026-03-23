from __future__ import annotations

from typing import Any

import app.config as config
from app.db.connection import get_cursor
from app.db.utils import normalize_row_json_fields, normalize_rows_json_fields, safe_json_dumps


CHUNK_JSON_FIELDS = ("metadata_json", "section_path", "embedding")


def _column_exists(cursor, table_name: str, column_name: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
        LIMIT 1
        """,
        (table_name, column_name),
    )
    return cursor.fetchone() is not None


def _chunk_text_column(cursor) -> str:
    if _column_exists(cursor, "document_chunks", "chunk_text"):
        return "chunk_text"
    if _column_exists(cursor, "document_chunks", "content"):
        return "content"
    raise RuntimeError("document_chunks table has neither 'chunk_text' nor 'content' column")


def _normalize_chunk_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    row = normalize_row_json_fields(row, CHUNK_JSON_FIELDS)
    if row is None:
        return None

    normalized = dict(row)

    if "chunk_text" not in normalized:
        normalized["chunk_text"] = normalized.get("content", "")
    if "content" not in normalized:
        normalized["content"] = normalized.get("chunk_text", "")

    if "metadata_json" not in normalized:
        normalized["metadata_json"] = normalized.get("metadata") or {}

    if "search_text" not in normalized:
        normalized["search_text"] = normalized.get("chunk_text", "")
    if "lexical_text" not in normalized:
        normalized["lexical_text"] = normalized.get("search_text", "") or normalized.get("chunk_text", "")

    return normalized


def _normalize_chunk_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_chunk_row(row) for row in rows]


def insert_chunk(
    document_id: int,
    chunk_text: str,
    embedding: Any = None,
    chunk_index: int | None = None,
    section_path: Any = None,
    page_start: int | None = None,
    page_end: int | None = None,
    block_start_index: int | None = None,
    block_end_index: int | None = None,
    chunk_type: str | None = None,
    metadata_json: dict[str, Any] | list[Any] | str | None = None,
    search_text: str | None = None,
    lexical_text: str | None = None,
    doc_title: str | None = None,
    section_title: str | None = None,
    token_count: int | None = None,
    chunk_hash: str | None = None,
    embedding_model: str | None = None,
) -> int:
    if embedding_model is None:
        embedding_model = getattr(config, "OLLAMA_EMBED_MODEL", None)

    with get_cursor(commit=True) as (_, cursor):
        text_column = _chunk_text_column(cursor)

        columns = ["document_id", text_column]
        values: list[Any] = [document_id, chunk_text]

        def add_if_exists(column_name: str, value: Any, *, json_dump: bool = False) -> None:
            if _column_exists(cursor, "document_chunks", column_name):
                columns.append(column_name)
                values.append(safe_json_dumps(value) if json_dump else value)

        add_if_exists("embedding", embedding, json_dump=True)
        add_if_exists("chunk_index", chunk_index)
        add_if_exists("section_path", section_path, json_dump=not isinstance(section_path, str))
        add_if_exists("page_start", page_start)
        add_if_exists("page_end", page_end)
        add_if_exists("block_start_index", block_start_index)
        add_if_exists("block_end_index", block_end_index)
        add_if_exists("chunk_type", chunk_type)
        add_if_exists("metadata_json", metadata_json, json_dump=not isinstance(metadata_json, str))
        add_if_exists("search_text", search_text)
        add_if_exists("lexical_text", lexical_text)
        add_if_exists("doc_title", doc_title)
        add_if_exists("section_title", section_title)
        add_if_exists("token_count", token_count)
        add_if_exists("chunk_hash", chunk_hash)
        add_if_exists("embedding_model", embedding_model)

        placeholders = ", ".join(["%s"] * len(columns))
        column_sql = ", ".join(columns)

        unique_on_document_chunk = (
            chunk_index is not None
            and _column_exists(cursor, "document_chunks", "document_id")
            and _column_exists(cursor, "document_chunks", "chunk_index")
        )

        if unique_on_document_chunk:
            update_columns = [c for c in columns if c not in {"document_id", "chunk_index"}]
            update_sql = ", ".join(f"{c} = VALUES({c})" for c in update_columns)

            if _column_exists(cursor, "document_chunks", "updated_at"):
                if update_sql:
                    update_sql += ", updated_at = CURRENT_TIMESTAMP"
                else:
                    update_sql = "updated_at = CURRENT_TIMESTAMP"

            cursor.execute(
                f"""
                INSERT INTO document_chunks ({column_sql})
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE {update_sql}
                """,
                values,
            )

            cursor.execute(
                """
                SELECT id
                FROM document_chunks
                WHERE document_id = %s AND chunk_index = %s
                LIMIT 1
                """,
                (document_id, chunk_index),
            )
            row = cursor.fetchone()
            return int(row["id"])

        cursor.execute(
            f"""
            INSERT INTO document_chunks ({column_sql})
            VALUES ({placeholders})
            """,
            values,
        )
        return int(cursor.lastrowid)


def get_chunk_by_id(chunk_id: int) -> dict[str, Any] | None:
    with get_cursor() as (_, cursor):
        text_column = _chunk_text_column(cursor)

        if text_column == "content":
            select_sql = """
                SELECT
                    id,
                    document_id,
                    chunk_index,
                    content,
                    content AS chunk_text,
                    search_text,
                    lexical_text,
                    embedding,
                    chunk_type,
                    doc_title,
                    section_title,
                    section_path,
                    page_start,
                    page_end,
                    block_start_index,
                    block_end_index,
                    token_count,
                    chunk_hash,
                    metadata_json,
                    embedding_model,
                    created_at,
                    updated_at
                FROM document_chunks
                WHERE id = %s
                LIMIT 1
            """
        else:
            select_sql = """
                SELECT *
                FROM document_chunks
                WHERE id = %s
                LIMIT 1
            """

        cursor.execute(select_sql, (chunk_id,))
        row = cursor.fetchone()
        return _normalize_chunk_row(row)


def get_chunks_by_document_id(document_id: int) -> list[dict[str, Any]]:
    with get_cursor() as (_, cursor):
        text_column = _chunk_text_column(cursor)

        if text_column == "content":
            select_sql = """
                SELECT
                    id,
                    document_id,
                    chunk_index,
                    content,
                    content AS chunk_text,
                    search_text,
                    lexical_text,
                    embedding,
                    chunk_type,
                    doc_title,
                    section_title,
                    section_path,
                    page_start,
                    page_end,
                    block_start_index,
                    block_end_index,
                    token_count,
                    chunk_hash,
                    metadata_json,
                    embedding_model,
                    created_at,
                    updated_at
                FROM document_chunks
                WHERE document_id = %s
                ORDER BY chunk_index ASC, id ASC
            """
        else:
            select_sql = """
                SELECT *
                FROM document_chunks
                WHERE document_id = %s
                ORDER BY chunk_index ASC, id ASC
            """

        cursor.execute(select_sql, (document_id,))
        rows = cursor.fetchall() or []
        return _normalize_chunk_rows(rows)


def get_all_chunks() -> list[dict[str, Any]]:
    with get_cursor() as (_, cursor):
        text_column = _chunk_text_column(cursor)

        if text_column == "content":
            select_sql = """
                SELECT
                    id,
                    document_id,
                    chunk_index,
                    content,
                    content AS chunk_text,
                    search_text,
                    lexical_text,
                    embedding,
                    chunk_type,
                    doc_title,
                    section_title,
                    section_path,
                    page_start,
                    page_end,
                    block_start_index,
                    block_end_index,
                    token_count,
                    chunk_hash,
                    metadata_json,
                    embedding_model,
                    created_at,
                    updated_at
                FROM document_chunks
                ORDER BY document_id ASC, chunk_index ASC, id ASC
            """
        else:
            select_sql = """
                SELECT *
                FROM document_chunks
                ORDER BY document_id ASC, chunk_index ASC, id ASC
            """

        cursor.execute(select_sql)
        rows = cursor.fetchall() or []
        return _normalize_chunk_rows(rows)


def clear_chunks_by_document_id(document_id: int) -> int:
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute(
            """
            DELETE FROM document_chunks
            WHERE document_id = %s
            """,
            (document_id,),
        )
        return int(cursor.rowcount)


def clear_all_chunks() -> int:
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute("DELETE FROM document_chunks")
        return int(cursor.rowcount)