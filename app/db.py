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
        cursorclass=pymysql.cursors.Cursor,
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
        cursor.close()
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