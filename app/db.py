import json
from contextlib import contextmanager
from typing import Any, Iterable

import pymysql
from pymysql.cursors import DictCursor

import app.config as config


def _pick_attr(*names: str, default=None):
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None and str(value).strip() != "":
                return value
    return default


DB_HOST = _pick_attr("DB_HOST", "MYSQL_HOST", default="127.0.0.1")
DB_PORT = int(_pick_attr("DB_PORT", "MYSQL_PORT", default=3306))
DB_USER = _pick_attr("DB_USER", "MYSQL_USER", default="root")
DB_PASSWORD = _pick_attr("DB_PASSWORD", "MYSQL_PASSWORD", default="")
DB_NAME = _pick_attr("DB_NAME", "MYSQL_DATABASE", "MYSQL_DB", default="knowledge_base")


def get_connection():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
    )


@contextmanager
def get_cursor(commit: bool = False):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            yield conn, cursor
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _safe_json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _normalize_row_json_fields(row: dict[str, Any], json_fields: Iterable[str]) -> dict[str, Any]:
    if not row:
        return row
    normalized = dict(row)
    for field in json_fields:
        value = normalized.get(field)
        if value is None or value == "":
            normalized[field] = None
            continue
        if isinstance(value, (dict, list)):
            continue
        try:
            normalized[field] = json.loads(value)
        except Exception:
            pass
    return normalized


def init_db():
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                title VARCHAR(512) NOT NULL,
                content LONGTEXT,
                raw_text LONGTEXT,
                file_path VARCHAR(1024),
                file_type VARCHAR(64),
                source_type VARCHAR(64) DEFAULT 'upload',
                lang VARCHAR(32),
                author VARCHAR(255),
                published_at DATETIME NULL,
                content_hash VARCHAR(128),
                block_count INT DEFAULT 0,
                blocks_json LONGTEXT,
                metadata_json LONGTEXT,
                tags_json LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_documents_title (title(255)),
                INDEX idx_documents_file_type (file_type),
                INDEX idx_documents_source_type (source_type),
                INDEX idx_documents_content_hash (content_hash)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                document_id BIGINT NOT NULL,
                chunk_index INT NOT NULL,
                chunk_text LONGTEXT NOT NULL,
                search_text LONGTEXT,
                lexical_text LONGTEXT,
                embedding LONGTEXT,
                chunk_type VARCHAR(64),
                doc_title VARCHAR(512),
                section_title VARCHAR(512),
                section_path VARCHAR(1024),
                page_start INT NULL,
                page_end INT NULL,
                block_start_index INT NULL,
                block_end_index INT NULL,
                token_count INT DEFAULT 0,
                chunk_hash VARCHAR(128),
                metadata_json LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                CONSTRAINT fk_chunks_document
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                    ON DELETE CASCADE,
                UNIQUE KEY uniq_document_chunk_index (document_id, chunk_index),
                INDEX idx_chunks_document_id (document_id),
                INDEX idx_chunks_doc_title (doc_title(255)),
                INDEX idx_chunks_section_title (section_title(255)),
                INDEX idx_chunks_chunk_type (chunk_type),
                INDEX idx_chunks_chunk_hash (chunk_hash)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                session_id VARCHAR(128) NOT NULL,
                title VARCHAR(255),
                summary_text LONGTEXT,
                metadata_json LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_chat_session_id (session_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                session_id VARCHAR(128) NOT NULL,
                role VARCHAR(32) NOT NULL,
                message LONGTEXT NOT NULL,
                rewritten_query LONGTEXT NULL,
                sources_json LONGTEXT,
                metadata_json LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_chat_messages_session_id (session_id),
                INDEX idx_chat_messages_role (role),
                CONSTRAINT fk_chat_messages_session
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                    ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )

        _run_best_effort_migrations(cursor)


def _run_best_effort_migrations(cursor):
    migrations = [
        ("documents", "source_type", "ALTER TABLE documents ADD COLUMN source_type VARCHAR(64) DEFAULT 'upload'"),
        ("documents", "lang", "ALTER TABLE documents ADD COLUMN lang VARCHAR(32)"),
        ("documents", "author", "ALTER TABLE documents ADD COLUMN author VARCHAR(255)"),
        ("documents", "published_at", "ALTER TABLE documents ADD COLUMN published_at DATETIME NULL"),
        ("documents", "content_hash", "ALTER TABLE documents ADD COLUMN content_hash VARCHAR(128)"),
        ("documents", "tags_json", "ALTER TABLE documents ADD COLUMN tags_json LONGTEXT"),
        ("document_chunks", "search_text", "ALTER TABLE document_chunks ADD COLUMN search_text LONGTEXT"),
        ("document_chunks", "lexical_text", "ALTER TABLE document_chunks ADD COLUMN lexical_text LONGTEXT"),
        ("document_chunks", "doc_title", "ALTER TABLE document_chunks ADD COLUMN doc_title VARCHAR(512)"),
        ("document_chunks", "section_title", "ALTER TABLE document_chunks ADD COLUMN section_title VARCHAR(512)"),
        ("document_chunks", "token_count", "ALTER TABLE document_chunks ADD COLUMN token_count INT DEFAULT 0"),
        ("document_chunks", "chunk_hash", "ALTER TABLE document_chunks ADD COLUMN chunk_hash VARCHAR(128)"),
        (
            "chat_sessions",
            "session_id",
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                session_id VARCHAR(128) NOT NULL,
                title VARCHAR(255),
                summary_text LONGTEXT,
                metadata_json LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_chat_session_id (session_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
        ),
        (
            "chat_messages",
            "session_id",
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                session_id VARCHAR(128) NOT NULL,
                role VARCHAR(32) NOT NULL,
                message LONGTEXT NOT NULL,
                rewritten_query LONGTEXT NULL,
                sources_json LONGTEXT,
                metadata_json LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_chat_messages_session_id (session_id),
                INDEX idx_chat_messages_role (role),
                CONSTRAINT fk_chat_messages_session
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                    ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
        ),
    ]

    for table_name, column_name, sql in migrations:
        try:
            if table_name in {"chat_sessions", "chat_messages"}:
                cursor.execute(sql)
                continue

            cursor.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s AND column_name = %s
                """,
                (DB_NAME, table_name, column_name),
            )
            row = cursor.fetchone()
            if not row or row["cnt"] == 0:
                cursor.execute(sql)
        except Exception:
            pass


# -----------------------------
# documents
# -----------------------------
def insert_document(
    title: str,
    content: str,
    raw_text: str | None = None,
    file_path: str | None = None,
    file_type: str | None = None,
    source_type: str | None = "upload",
    lang: str | None = None,
    author: str | None = None,
    published_at: Any = None,
    content_hash: str | None = None,
    block_count: int = 0,
    blocks_json: str | dict | list | None = None,
    metadata_json: str | dict | list | None = None,
    tags_json: str | dict | list | None = None,
) -> int:
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute(
            """
            INSERT INTO documents (
                title, content, raw_text, file_path, file_type,
                source_type, lang, author, published_at, content_hash,
                block_count, blocks_json, metadata_json, tags_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                title,
                content,
                raw_text,
                file_path,
                file_type,
                source_type,
                lang,
                author,
                published_at,
                content_hash,
                block_count,
                _safe_json_dumps(blocks_json),
                _safe_json_dumps(metadata_json),
                _safe_json_dumps(tags_json),
            ),
        )
        return cursor.lastrowid


def update_document(document_id: int, **fields: Any) -> bool:
    if not fields:
        return False

    allowed = {
        "title",
        "content",
        "raw_text",
        "file_path",
        "file_type",
        "source_type",
        "lang",
        "author",
        "published_at",
        "content_hash",
        "block_count",
        "blocks_json",
        "metadata_json",
        "tags_json",
    }

    update_fields = []
    values = []

    for key, value in fields.items():
        if key not in allowed:
            continue
        if key in {"blocks_json", "metadata_json", "tags_json"}:
            value = _safe_json_dumps(value)
        update_fields.append(f"{key} = %s")
        values.append(value)

    if not update_fields:
        return False

    values.append(document_id)

    with get_cursor(commit=True) as (_, cursor):
        cursor.execute(
            f"""
            UPDATE documents
            SET {", ".join(update_fields)}
            WHERE id = %s
            """,
            values,
        )
        return cursor.rowcount > 0


def get_document_by_id(document_id: int) -> dict[str, Any] | None:
    with get_cursor() as (_, cursor):
        cursor.execute("SELECT * FROM documents WHERE id = %s", (document_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return _normalize_row_json_fields(row, ["blocks_json", "metadata_json", "tags_json"])


def get_all_documents() -> list[dict[str, Any]]:
    with get_cursor() as (_, cursor):
        cursor.execute("SELECT * FROM documents ORDER BY id DESC")
        rows = cursor.fetchall() or []
        return [_normalize_row_json_fields(r, ["blocks_json", "metadata_json", "tags_json"]) for r in rows]


def search_documents(keyword: str) -> list[dict[str, Any]]:
    q = f"%{keyword}%"
    with get_cursor() as (_, cursor):
        cursor.execute(
            """
            SELECT *
            FROM documents
            WHERE title LIKE %s OR content LIKE %s OR raw_text LIKE %s
            ORDER BY id DESC
            """,
            (q, q, q),
        )
        rows = cursor.fetchall() or []
        return [_normalize_row_json_fields(r, ["blocks_json", "metadata_json", "tags_json"]) for r in rows]


def delete_document(document_id: int) -> bool:
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute("DELETE FROM documents WHERE id = %s", (document_id,))
        return cursor.rowcount > 0


# -----------------------------
# chunks
# -----------------------------
def insert_chunk(
    document_id: int,
    chunk_text: str,
    embedding: str | list | None,
    chunk_index: int,
    section_path: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    block_start_index: int | None = None,
    block_end_index: int | None = None,
    chunk_type: str | None = None,
    metadata_json: str | dict | list | None = None,
    search_text: str | None = None,
    lexical_text: str | None = None,
    doc_title: str | None = None,
    section_title: str | None = None,
    token_count: int = 0,
    chunk_hash: str | None = None,
) -> int:
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute(
            """
            INSERT INTO document_chunks (
                document_id, chunk_text, search_text, lexical_text, embedding, chunk_index,
                section_path, page_start, page_end, block_start_index, block_end_index,
                chunk_type, doc_title, section_title, token_count, chunk_hash, metadata_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                chunk_text = VALUES(chunk_text),
                search_text = VALUES(search_text),
                lexical_text = VALUES(lexical_text),
                embedding = VALUES(embedding),
                section_path = VALUES(section_path),
                page_start = VALUES(page_start),
                page_end = VALUES(page_end),
                block_start_index = VALUES(block_start_index),
                block_end_index = VALUES(block_end_index),
                chunk_type = VALUES(chunk_type),
                doc_title = VALUES(doc_title),
                section_title = VALUES(section_title),
                token_count = VALUES(token_count),
                chunk_hash = VALUES(chunk_hash),
                metadata_json = VALUES(metadata_json),
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                document_id,
                chunk_text,
                search_text,
                lexical_text,
                _safe_json_dumps(embedding),
                chunk_index,
                section_path,
                page_start,
                page_end,
                block_start_index,
                block_end_index,
                chunk_type,
                doc_title,
                section_title,
                token_count,
                chunk_hash,
                _safe_json_dumps(metadata_json),
            ),
        )
        return cursor.lastrowid


def get_chunks_by_document_id(document_id: int) -> list[dict[str, Any]]:
    with get_cursor() as (_, cursor):
        cursor.execute(
            """
            SELECT *
            FROM document_chunks
            WHERE document_id = %s
            ORDER BY chunk_index ASC
            """,
            (document_id,),
        )
        rows = cursor.fetchall() or []
        return [_normalize_row_json_fields(r, ["embedding", "metadata_json"]) for r in rows]


def get_all_chunks() -> list[dict[str, Any]]:
    with get_cursor() as (_, cursor):
        cursor.execute(
            """
            SELECT *
            FROM document_chunks
            ORDER BY document_id ASC, chunk_index ASC
            """
        )
        rows = cursor.fetchall() or []
        return [_normalize_row_json_fields(r, ["embedding", "metadata_json"]) for r in rows]


def get_chunk_by_id(chunk_id: int) -> dict[str, Any] | None:
    with get_cursor() as (_, cursor):
        cursor.execute("SELECT * FROM document_chunks WHERE id = %s", (chunk_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return _normalize_row_json_fields(row, ["embedding", "metadata_json"])


def clear_chunks_by_document_id(document_id: int) -> int:
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
        return cursor.rowcount


def clear_all_chunks() -> int:
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute("DELETE FROM document_chunks")
        return cursor.rowcount


# -----------------------------
# chat sessions / messages
# -----------------------------
def create_chat_session(
    session_id: str,
    title: str | None = None,
    summary_text: str | None = None,
    metadata_json: str | dict | list | None = None,
) -> str:
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute(
            """
            INSERT INTO chat_sessions (session_id, title, summary_text, metadata_json)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                title = COALESCE(VALUES(title), title),
                summary_text = COALESCE(VALUES(summary_text), summary_text),
                metadata_json = COALESCE(VALUES(metadata_json), metadata_json),
                updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, title, summary_text, _safe_json_dumps(metadata_json)),
        )
    return session_id


def get_chat_session(session_id: str) -> dict[str, Any] | None:
    with get_cursor() as (_, cursor):
        cursor.execute("SELECT * FROM chat_sessions WHERE session_id = %s", (session_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return _normalize_row_json_fields(row, ["metadata_json"])


def list_chat_sessions(limit: int = 50) -> list[dict[str, Any]]:
    with get_cursor() as (_, cursor):
        cursor.execute(
            """
            SELECT *
            FROM chat_sessions
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cursor.fetchall() or []
        return [_normalize_row_json_fields(r, ["metadata_json"]) for r in rows]


def update_chat_session(
    session_id: str,
    title: str | None = None,
    summary_text: str | None = None,
    metadata_json: str | dict | list | None = None,
) -> bool:
    updates = []
    values: list[Any] = []

    if title is not None:
        updates.append("title = %s")
        values.append(title)
    if summary_text is not None:
        updates.append("summary_text = %s")
        values.append(summary_text)
    if metadata_json is not None:
        updates.append("metadata_json = %s")
        values.append(_safe_json_dumps(metadata_json))

    if not updates:
        return False

    updates.append("updated_at = CURRENT_TIMESTAMP")
    values.append(session_id)

    with get_cursor(commit=True) as (_, cursor):
        cursor.execute(
            f"""
            UPDATE chat_sessions
            SET {", ".join(updates)}
            WHERE session_id = %s
            """,
            values,
        )
        return cursor.rowcount > 0


def insert_chat_message(
    session_id: str,
    role: str,
    message: str,
    rewritten_query: str | None = None,
    sources_json: str | dict | list | None = None,
    metadata_json: str | dict | list | None = None,
) -> int:
    create_chat_session(session_id=session_id)

    with get_cursor(commit=True) as (_, cursor):
        cursor.execute(
            """
            INSERT INTO chat_messages (
                session_id, role, message, rewritten_query, sources_json, metadata_json
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                session_id,
                role,
                message,
                rewritten_query,
                _safe_json_dumps(sources_json),
                _safe_json_dumps(metadata_json),
            ),
        )
        return cursor.lastrowid


def get_chat_messages(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    with get_cursor() as (_, cursor):
        cursor.execute(
            """
            SELECT *
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (session_id, limit),
        )
        rows = cursor.fetchall() or []
        rows.reverse()
        return [_normalize_row_json_fields(r, ["sources_json", "metadata_json"]) for r in rows]


def delete_chat_session(session_id: str) -> bool:
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
        return cursor.rowcount > 0