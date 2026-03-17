import logging

from fastapi import FastAPI, HTTPException

from app.db import (
    get_all_documents,
    search_documents,
    get_document_by_id,
    init_db,
)
from app.models import (
    DocumentResponse,
    ImportRequest,
    ImportResponse,
    IndexRequest,
    IndexResponse,
    RetrieveRequest,
    RetrieveResponse,
    AskRequest,
    AskResponse,
    ChunkResponse,
)
from app.services import (
    import_documents,
    summarize_text,
    index_document,
    retrieve_chunks,
    answer_question,
    get_document_chunks,
)
from app.utils import setup_logger

setup_logger()

app = FastAPI(title="Knowledge Base RAG API")


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/")
def home():
    logging.info("访问首页接口")
    return {"message": "Knowledge Base RAG API is running"}


@app.get("/documents")
def list_documents():
    docs = get_all_documents()
    return [
        DocumentResponse(
            id=row["id"],
            title=row["title"],
            file_path=row["file_path"],
            created_at=str(row["created_at"]) if row["created_at"] else None,
        )
        for row in docs
    ]


@app.get("/search")
def search(q: str):
    docs = search_documents(q)
    return {
        "results": [
            DocumentResponse(
                id=row["id"],
                title=row["title"],
                file_path=row["file_path"],
                created_at=str(row["created_at"]) if row["created_at"] else None,
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
        id=row["id"],
        title=row["title"],
        content=row["content"],
        file_path=row["file_path"],
        created_at=str(row["created_at"]) if row["created_at"] else None,
    )


@app.post("/documents/import", response_model=ImportResponse)
def import_docs(payload: ImportRequest):
    try:
        doc_ids = import_documents(payload.folder)
        return ImportResponse(
            imported_count=len(doc_ids),
            document_ids=doc_ids,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/{doc_id}/summary")
def summarize_document(doc_id: int):
    row = get_document_by_id(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="document not found")

    try:
        summary = summarize_text(row["content"])
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/index/{doc_id}", response_model=IndexResponse)
def rag_index_document(doc_id: int, payload: IndexRequest):
    row = get_document_by_id(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="document not found")

    try:
        result = index_document(
            doc_id=doc_id,
            chunk_size=payload.chunk_size,
            overlap=payload.overlap,
        )
        return IndexResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunks/{doc_id}", response_model=list[ChunkResponse])
def list_document_chunks(doc_id: int):
    row = get_document_by_id(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="document not found")

    try:
        chunks = get_document_chunks(doc_id)
        return [ChunkResponse(**chunk) for chunk in chunks]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/retrieve", response_model=RetrieveResponse)
def rag_retrieve(payload: RetrieveRequest):
    try:
        results = retrieve_chunks(query=payload.query, top_k=payload.top_k)
        return RetrieveResponse(
            query=payload.query,
            results=[ChunkResponse(**item) for item in results],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/ask", response_model=AskResponse)
def rag_ask(payload: AskRequest):
    try:
        result = answer_question(query=payload.query, top_k=payload.top_k)
        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[ChunkResponse(**item) for item in result["sources"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))