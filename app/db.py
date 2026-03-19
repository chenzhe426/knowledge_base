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
        autocommit=False,
    )


def _column_exists(cursor, table_name: str, column_name: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
        LIMIT 1
        """,
        (MYSQL_DATABASE, table_name, column_name),
    )
    return cursor.fetchone() is not None


def _index_exists(cursor, table_name: str, index_name: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = %s
          AND INDEX_NAME = %s
        LIMIT 1
        """,
        (MYSQL_DATABASE, table_name, index_name),
    )
    return cursor.fetchone() is not None


def init_db():
    """
    初始化数据库，并对旧表做向后兼容升级：
    - documents 增加 file_type / source_type / metadata_json / block_count / blocks_json / raw_text
    - document_chunks 增加 chunk_type / section_title / section_path_json / page_start / page_end / block_start_order / block_end_order / token_count / metadata_json
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    content LONGTEXT NOT NULL,
                    raw_text LONGTEXT NULL,
                    file_path VARCHAR(1000) NULL,
                    file_type VARCHAR(50) NULL,
                    source_type VARCHAR(50) NULL,
                    metadata_json LONGTEXT NULL,
                    block_count INT NOT NULL DEFAULT 0,
                    blocks_json LONGTEXT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            # 向旧 documents 表补字段
            document_columns = {
                "raw_text": "ALTER TABLE documents ADD COLUMN raw_text LONGTEXT NULL AFTER content",
                "file_type": "ALTER TABLE documents ADD COLUMN file_type VARCHAR(50) NULL AFTER file_path",
                "source_type": "ALTER TABLE documents ADD COLUMN source_type VARCHAR(50) NULL AFTER file_type",
                "metadata_json": "ALTER TABLE documents ADD COLUMN metadata_json LONGTEXT NULL AFTER source_type",
                "block_count": "ALTER TABLE documents ADD COLUMN block_count INT NOT NULL DEFAULT 0 AFTER metadata_json",
                "blocks_json": "ALTER TABLE documents ADD COLUMN blocks_json LONGTEXT NULL AFTER block_count",
            }
            for column_name, ddl in document_columns.items():
                if not _column_exists(cursor, "documents", column_name):
                    cursor.execute(ddl)

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    document_id INT NOT NULL,
                    chunk_index INT NOT NULL,
                    chunk_text LONGTEXT NOT NULL,
                    embedding LONGTEXT NOT NULL,
                    chunk_type VARCHAR(50) NOT NULL DEFAULT 'paragraph',
                    section_title VARCHAR(255) NULL,
                    section_path_json LONGTEXT NULL,
                    char_start INT NOT NULL DEFAULT 0,
                    char_end INT NOT NULL DEFAULT 0,
                    page_start INT NULL,
                    page_end INT NULL,
                    block_start_order INT NULL,
                    block_end_order INT NULL,
                    token_count INT NOT NULL DEFAULT 0,
                    metadata_json LONGTEXT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_document_chunks_document
                        FOREIGN KEY (document_id) REFERENCES documents(id)
                        ON DELETE CASCADE,
                    UNIQUE KEY uniq_document_chunk (document_id, chunk_index),
                    INDEX idx_document_id (document_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            # 向旧 document_chunks 表补字段
            chunk_columns = {
                "chunk_type": "ALTER TABLE document_chunks ADD COLUMN chunk_type VARCHAR(50) NOT NULL DEFAULT 'paragraph' AFTER embedding",
                "section_title": "ALTER TABLE document_chunks ADD COLUMN section_title VARCHAR(255) NULL AFTER chunk_type",
                "section_path_json": "ALTER TABLE document_chunks ADD COLUMN section_path_json LONGTEXT NULL AFTER section_title",
                "page_start": "ALTER TABLE document_chunks ADD COLUMN page_start INT NULL AFTER char_end",
                "page_end": "ALTER TABLE document_chunks ADD COLUMN page_end INT NULL AFTER page_start",
                "block_start_order": "ALTER TABLE document_chunks ADD COLUMN block_start_order INT NULL AFTER page_end",
                "block_end_order": "ALTER TABLE document_chunks ADD COLUMN block_end_order INT NULL AFTER block_start_order",
                "token_count": "ALTER TABLE document_chunks ADD COLUMN token_count INT NOT NULL DEFAULT 0 AFTER block_end_order",
                "metadata_json": "ALTER TABLE document_chunks ADD COLUMN metadata_json LONGTEXT NULL AFTER token_count",
            }
            for column_name, ddl in chunk_columns.items():
                if not _column_exists(cursor, "document_chunks", column_name):
                    cursor.execute(ddl)

            if not _index_exists(cursor, "document_chunks", "idx_document_section_title"):
                cursor.execute(
                    """
                    CREATE INDEX idx_document_section_title
                    ON document_chunks (document_id, section_title)
                    """
                )

            conn.commit()
    finally:
        conn.close()


def insert_document(
    title: str,
    content: str,
    raw_text: str | None = None,
    file_path: str | None = None,
    file_type: str | None = None,
    source_type: str | None = None,
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
                    title,
                    content,
                    raw_text,
                    file_path,
                    file_type,
                    source_type,
                    metadata_json,
                    block_count,
                    blocks_json
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
                    json.dumps(metadata or {}, ensure_ascii=False),
                    block_count,
                    json.dumps(blocks or [], ensure_ascii=False),
                ),
            )
            doc_id = cursor.lastrowid
            conn.commit()
            return doc_id
    finally:
        conn.close()


def update_document_blocks(
    doc_id: int,
    content: str,
    raw_text: str | None,
    metadata: dict[str, Any] | None,
    block_count: int,
    blocks: list[dict[str, Any]] | None,
) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE documents
                SET content = %s,
                    raw_text = %s,
                    metadata_json = %s,
                    block_count = %s,
                    blocks_json = %s
                WHERE id = %s
                """,
                (
                    content,
                    raw_text,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    block_count,
                    json.dumps(blocks or [], ensure_ascii=False),
                    doc_id,
                ),
            )
            conn.commit()
    finally:
        conn.close()


def get_all_documents() -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    title,
                    file_path,
                    file_type,
                    source_type,
                    block_count,
                    created_at
                FROM documents
                ORDER BY id DESC
                """
            )
            return cursor.fetchall()
    finally:
        conn.close()


def get_document_by_id(doc_id: int) -> dict[str, Any] | None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    title,
                    content,
                    raw_text,
                    file_path,
                    file_type,
                    source_type,
                    metadata_json,
                    block_count,
                    blocks_json,
                    created_at
                FROM documents
                WHERE id = %s
                """,
                (doc_id,),
            )
            return cursor.fetchone()
    finally:
        conn.close()


def search_documents(query: str) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        keyword = f"%{query}%"
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    title,
                    file_path,
                    file_type,
                    source_type,
                    block_count,
                    created_at
                FROM documents
                WHERE title LIKE %s
                   OR content LIKE %s
                ORDER BY id DESC
                """,
                (keyword, keyword),
            )
            return cursor.fetchall()
    finally:
        conn.close()


def insert_chunk(
    document_id: int,
    chunk_index: int,
    chunk_text: str,
    embedding: list[float],
    char_start: int = 0,
    char_end: int = 0,
    chunk_type: str = "paragraph",
    section_title: str | None = None,
    section_path: list[str] | None = None,
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
                    document_id,
                    chunk_index,
                    chunk_text,
                    embedding,
                    chunk_type,
                    section_title,
                    section_path_json,
                    char_start,
                    char_end,
                    page_start,
                    page_end,
                    block_start_order,
                    block_end_order,
                    token_count,
                    metadata_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    chunk_text = VALUES(chunk_text),
                    embedding = VALUES(embedding),
                    chunk_type = VALUES(chunk_type),
                    section_title = VALUES(section_title),
                    section_path_json = VALUES(section_path_json),
                    char_start = VALUES(char_start),
                    char_end = VALUES(char_end),
                    page_start = VALUES(page_start),
                    page_end = VALUES(page_end),
                    block_start_order = VALUES(block_start_order),
                    block_end_order = VALUES(block_end_order),
                    token_count = VALUES(token_count),
                    metadata_json = VALUES(metadata_json)
                """,
                (
                    document_id,
                    chunk_index,
                    chunk_text,
                    json.dumps(embedding, ensure_ascii=False),
                    chunk_type,
                    section_title,
                    json.dumps(section_path or [], ensure_ascii=False),
                    char_start,
                    char_end,
                    page_start,
                    page_end,
                    block_start_order,
                    block_end_order,
                    token_count,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            chunk_id = cursor.lastrowid
            conn.commit()
            return chunk_id
    finally:
        conn.close()


def clear_chunks_by_document_id(document_id: int) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM document_chunks
                WHERE document_id = %s
                """,
                (document_id,),
            )
            conn.commit()
    finally:
        conn.close()


def get_chunks_by_document_id(document_id: int) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    c.id,
                    c.document_id,
                    d.title,
                    c.chunk_index,
                    c.chunk_text,
                    c.chunk_type,
                    c.section_title,
                    c.section_path_json,
                    c.char_start,
                    c.char_end,
                    c.page_start,
                    c.page_end,
                    c.block_start_order,
                    c.block_end_order,
                    c.token_count,
                    c.metadata_json,
                    c.created_at
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = %s
                ORDER BY c.chunk_index ASC
                """,
                (document_id,),
            )
            return cursor.fetchall()
    finally:
        conn.close()


def get_all_chunks() -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    c.id,
                    c.document_id,
                    d.title,
                    c.chunk_index,
                    c.chunk_text,
                    c.embedding,
                    c.chunk_type,
                    c.section_title,
                    c.section_path_json,
                    c.char_start,
                    c.char_end,
                    c.page_start,
                    c.page_end,
                    c.block_start_order,
                    c.block_end_order,
                    c.token_count,
                    c.metadata_json,
                    c.created_at
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY c.id ASC
                """
            )
            return cursor.fetchall()
    finally:
        conn.close()


def delete_document(doc_id: int) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM documents
                WHERE id = %s
                """,
                (doc_id,),
            )
            conn.commit()
    finally:
        conn.close()