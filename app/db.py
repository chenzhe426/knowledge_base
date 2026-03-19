import json
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from app.config import (
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
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


def init_db():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    content LONGTEXT NOT NULL,
                    file_path VARCHAR(1000) NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    document_id INT NOT NULL,
                    chunk_index INT NOT NULL,
                    chunk_text LONGTEXT NOT NULL,
                    embedding LONGTEXT NOT NULL,
                    char_start INT NOT NULL DEFAULT 0,
                    char_end INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_document_chunks_document
                        FOREIGN KEY (document_id) REFERENCES documents(id)
                        ON DELETE CASCADE,
                    UNIQUE KEY uniq_document_chunk (document_id, chunk_index),
                    INDEX idx_document_id (document_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

        conn.commit()
    finally:
        conn.close()


def insert_document(title: str, content: str, file_path: str | None = None) -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (title, content, file_path)
                VALUES (%s, %s, %s)
                """,
                (title, content, file_path),
            )
            doc_id = cursor.lastrowid
        conn.commit()
        return doc_id
    finally:
        conn.close()


def get_all_documents() -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, title, file_path, created_at
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
                SELECT id, title, content, file_path, created_at
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
                SELECT id, title, file_path, created_at
                FROM documents
                WHERE title LIKE %s OR content LIKE %s
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
                    char_start,
                    char_end
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    chunk_text = VALUES(chunk_text),
                    embedding = VALUES(embedding),
                    char_start = VALUES(char_start),
                    char_end = VALUES(char_end)
                """,
                (
                    document_id,
                    chunk_index,
                    chunk_text,
                    json.dumps(embedding, ensure_ascii=False),
                    char_start,
                    char_end,
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
                    c.char_start,
                    c.char_end,
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
                    c.char_start,
                    c.char_end,
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