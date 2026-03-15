import sqlite3


DB_NAME = "knowledge.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        file_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def insert_document(title: str, content: str, file_path: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO documents (title, content, file_path)
    VALUES (?, ?, ?)
    """, (title, content, file_path))

    conn.commit()
    conn.close()
def get_all_documents():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, title, file_path, created_at FROM documents")
    rows = cursor.fetchall()

    conn.close()
    return rows

def search_documents(keyword: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, title, file_path, created_at
    FROM documents
    WHERE title LIKE ? OR content LIKE ?
    """, (f"%{keyword}%", f"%{keyword}%"))

    rows = cursor.fetchall()
    conn.close()
    return rows

  