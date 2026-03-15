from fastapi import FastAPI
from app.db import get_all_documents, search_documents, get_connection
from app.services import import_documents
from app.services import summarize_text
app = FastAPI()


@app.get("/")
def home():
    return {"message": "Knowledge Base API is running"}


@app.get("/documents")
def list_documents():
    docs = get_all_documents()
    return {"documents": docs}


@app.get("/search")
def search(q: str):
    docs = search_documents(q)
    return {"results": docs}


@app.get("/documents/{doc_id}")
def get_document(doc_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, content, file_path, created_at FROM documents WHERE id = ?",
        (doc_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return {"document": row}
    return {"error": "document not found"}

@app.post("/documents/{doc_id}/summary")
def summarize_document(doc_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {"error": "document not found"}

    summary = summarize_text(row[0])
    return {"summary": summary}