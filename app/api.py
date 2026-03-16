from app.db import get_all_documents, search_documents, get_connection
from app.services import import_documents
from app.services import summarize_text
from app.utils import setup_logger
from app.models import DocumentResponse
from fastapi import FastAPI, HTTPException
import logging

setup_logger()

app = FastAPI()


@app.get("/")
def home():
    logging.info("访问首页接口")
    return {"message": "Knowledge Base API is running"}


@app.get("/documents")
def list_documents():
    docs = get_all_documents()
    return [
        DocumentResponse(
            id=row[0],
            title=row[1],
            file_path=row[2],
            created_at=row[3]
        ) for row in docs
    ]


@app.get("/search")
def search(q: str):
    docs = search_documents(q)
    return {"results": docs}


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, content, file_path, created_at FROM documents WHERE id = ?",
        (doc_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="document not found") 

    return DocumentResponse(
        id=row[0],
        title=row[1],
        content=row[2],
        file_path=row[3],
        created_at=row[4]
    )

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