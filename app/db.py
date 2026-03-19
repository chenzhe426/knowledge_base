from __future__ import annotations

import json
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from app.config import (
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
)


def get_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
    )


def init_db():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    title VARCHAR(500) NOT NULL,
                    content LONGTEXT,
                    raw_text LONGTEXT,
                    file_path VARCHAR(1000),
                    file_type VARCHAR(50),
                    source_type VARCHAR(50) DEFAULT 'upload',
                    metadata_json JSON NULL,
                    block_count INT DEFAULT 0,
                    blocks_json LONGTEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    document_id BIGINT NOT NULL,
                    chunk_text LONGTEXT NOT NULL,
                    embedding LONGTEXT NULL,
                    chunk_type VARCHAR(50) DEFAULT 'paragraph',
                    section_title VARCHAR(500) NULL,
                    section_path VARCHAR(1000) NULL,
                    char_start INT DEFAULT 0,
                    char_end INT DEFAULT 0,
                    page_start INT NULL,
                    page_end INT NULL,
                    block_start_order INT NULL,
                    block_end_order INT NULL,
                    token_count INT DEFAULT 0,
                    metadata_json LONGTEXT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_document_chunks_document
                        FOREIGN KEY (document_id) REFERENCES documents(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
    finally:
        conn.close()


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: Any, default: Any):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return default
    text = value.strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _hydrate_document_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["metadata"] = _json_loads(row.pop("metadata_json", None), {})
    row["blocks"] = _json_loads(row.pop("blocks_json", None), [])
    return row


def _hydrate_chunk_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["metadata"] = _json_loads(row.pop("metadata_json", None), {})
    row["section_path"] = _json_loads(row.get("section_path"), [])
    row["embedding"] = _json_loads(row.get("embedding"), None)
    return row


def insert_document(
    *,
    title: str,
    content: str,
    raw_text: str | None = None,
    file_path: str | None = None,
    file_type: str | None = None,
    source_type: str = "upload",
    metadata: dict[str, Any] | None = None,
    block_count: int = 0,
    blocks: list[dict[str, Any]] | None = None,
) -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (
                    title, content, raw_text, file_path, file_type,
                    source_type, metadata_json, block_count, blocks_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    title,
                    content,
                    raw_text,
                    file_path,
                    file_type,
                    source_type,
                    _json_dumps(metadata or {}),
                    block_count,
                    _json_dumps(blocks or []),
                ),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_document_by_id(doc_id: int) -> dict[str, Any] | None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
            row = cursor.fetchone()
            return _hydrate_document_row(row) if row else None
    finally:
        conn.close()


def get_all_documents() -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM documents ORDER BY id DESC")
            rows = cursor.fetchall()
            return [_hydrate_document_row(row) for row in rows]
    finally:
        conn.close()


def search_documents(keyword: str) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            like = f"%{keyword}%"
            cursor.execute(
                """
                SELECT * FROM documents
                WHERE title LIKE %s OR content LIKE %s
                ORDER BY id DESC
                """,
                (like, like),
            )
            rows = cursor.fetchall()
            return [_hydrate_document_row(row) for row in rows]
    finally:
        conn.close()


def clear_chunks_by_document_id(doc_id: int):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM document_chunks WHERE document_id = %s",
                (doc_id,),
            )
    finally:
        conn.close()


def insert_chunk(
    *,
    document_id: int,
    chunk_text: str,
    embedding: list[float] | None = None,
    chunk_type: str = "paragraph",
    section_title: str | None = None,
    section_path: list[str] | None = None,
    char_start: int = 0,
    char_end: int = 0,
    page_start: int | None = None,
    page_end: int | None = None,
    block_start_order: int | None = None,
    block_end_order: int | None = None,
    token_count: int = 0,
    metadata: dict[str, Any] | None = None,
) -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO document_chunks (
                    document_id, chunk_text, embedding, chunk_type,
                    section_title, section_path, char_start, char_end,
                    page_start, page_end, block_start_order, block_end_order,
                    token_count, metadata_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    chunk_text,
                    _json_dumps(embedding),
                    chunk_type,
                    section_title,
                    _json_dumps(section_path or []),
                    char_start,
                    char_end,
                    page_start,
                    page_end,
                    block_start_order,
                    block_end_order,
                    token_count,
                    _json_dumps(metadata or {}),
                ),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_chunks_by_document_id(doc_id: int) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM document_chunks
                WHERE document_id = %s
                ORDER BY id ASC
                """,
                (doc_id,),
            )
            rows = cursor.fetchall()
            return [_hydrate_chunk_row(row) for row in rows]
    finally:
        conn.close()


def get_all_chunks() -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM document_chunks ORDER BY id ASC")
            rows = cursor.fetchall()
            return [_hydrate_chunk_row(row) for row in rows]
    finally:
        conn.close()