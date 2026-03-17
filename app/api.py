from fastapi import FastAPI, HTTPException
import logging
from app.db import get_all_documents, search_documents, get_document_by_id, init_db
from app.services import import_documents, summarize_text
from app.utils import setup_logger
from app.models import DocumentResponse

setup_logger()

app = FastAPI()
@app.on_event("startup")
def startup_event():
    init_db()


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
            created_at=str(row[3]) if row[3] else None
        )
        for row in docs
    ]


@app.get("/search")
def search(q: str):
    docs = search_documents(q)
    return {
        "results": [
            DocumentResponse(
                id=row[0],
                title=row[1],
                file_path=row[2],
                created_at=str(row[3]) if row[3] else None
            )
            for row in docs
        ]
    }


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: int):
    row = get_document_by_id(doc_id)

    if not row:
        raise HTTPException(status_code=404, detail="document not found")

    return DocumentResponse(
        id=row[0],
        title=row[1],
        content=row[2],
        file_path=row[3],
        created_at=str(row[4]) if row[4] else None
    )


@app.post("/documents/{doc_id}/summary")
def summarize_document(doc_id: int):
    row = get_document_by_id(doc_id)

    if not row:
        raise HTTPException(status_code=404, detail="document not found")

    summary = summarize_text(row[2])
    return {"summary": summary}