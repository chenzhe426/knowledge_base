import json
import pymysql

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
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def init_db():
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INT PRIMARY KEY AUTO_INCREMENT,
                title VARCHAR(255) NOT NULL,
                content LONGTEXT NOT NULL,
                file_path VARCHAR(500) UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INT PRIMARY KEY AUTO_INCREMENT,
                document_id INT NOT NULL,
                chunk_index INT NOT NULL,
                chunk_text LONGTEXT NOT NULL,
                embedding LONGTEXT NOT NULL,
                char_start INT DEFAULT 0,
                char_end INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_document_chunk (document_id, chunk_index),
                INDEX idx_document_id (document_id),
                CONSTRAINT fk_document_chunks_document
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                    ON DELETE CASCADE
            )
            """
        )

        cursor.close()
    finally:
        conn.close()


def insert_document(title: str, content: str, file_path: str):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO documents (title, content, file_path)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                title = VALUES(title),
                content = VALUES(content)
            """,
            (title, content, file_path),
        )

        if cursor.lastrowid:
            doc_id = cursor.lastrowid
        else:
            cursor.execute(
                "SELECT id FROM documents WHERE file_path = %s",
                (file_path,),
            )
            row = cursor.fetchone()
            doc_id = row["id"] if row else None

        cursor.close()
        return doc_id
    finally:
        conn.close()


def get_all_documents(limit: int = 10, offset: int = 0):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, title, file_path, created_at
            FROM documents
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        )
        rows = cursor.fetchall()
        cursor.close()
        return rows
    finally:
        conn.close()


def get_document_by_id(doc_id: int):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, title, content, file_path, created_at
            FROM documents
            WHERE id = %s
            """,
            (doc_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        return row
    finally:
        conn.close()


def search_documents(keyword: str, limit: int = 10, offset: int = 0):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        like_keyword = f"%{keyword}%"
        cursor.execute(
            """
            SELECT id, title, file_path, created_at
            FROM documents
            WHERE title LIKE %s OR content LIKE %s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """,
            (like_keyword, like_keyword, limit, offset),
        )
        rows = cursor.fetchall()
        cursor.close()
        return rows
    finally:
        conn.close()


def clear_chunks_by_document_id(document_id: int):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM document_chunks WHERE document_id = %s",
            (document_id,),
        )
        cursor.close()
    finally:
        conn.close()


def insert_chunk(
    document_id: int,
    chunk_index: int,
    chunk_text: str,
    embedding: list[float],
    char_start: int = 0,
    char_end: int = 0,
):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO document_chunks
                (document_id, chunk_index, chunk_text, embedding, char_start, char_end)
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
        cursor.close()
        return chunk_id
    finally:
        conn.close()


def get_chunks_by_document_id(document_id: int):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                dc.id,
                dc.document_id,
                d.title,
                dc.chunk_index,
                dc.chunk_text,
                dc.embedding,
                dc.char_start,
                dc.char_end,
                dc.created_at
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.document_id = %s
            ORDER BY dc.chunk_index ASC
            """,
            (document_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
        return rows
    finally:
        conn.close()


def get_all_chunks():
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                dc.id,
                dc.document_id,
                d.title,
                dc.chunk_index,
                dc.chunk_text,
                dc.embedding,
                dc.char_start,
                dc.char_end,
                dc.created_at
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            ORDER BY dc.document_id ASC, dc.chunk_index ASC
            """
        )
        rows = cursor.fetchall()
        cursor.close()
        return rows
    finally:
        conn.close()