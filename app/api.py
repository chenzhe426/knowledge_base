from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.db import init_db
from app.models import (
    AskRequest,
    AskResponse,
    GenericResponse,
    ImportFileRequest,
    ImportFolderRequest,
    IndexRequest,
    SummaryRequest,
)
from app.services import (
    answer_question,
    get_document_chunks,
    import_documents,
    import_single_document,
    index_document,
    list_documents,
    summarize_text,
)

app = FastAPI(title="knowledge_base", version="1.0.0")

@app.get("/")
def root():
    return {"message": "knowledge_base api is running"}


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/documents", response_model=GenericResponse)
def api_list_documents():
    return GenericResponse(data=list_documents())


@app.get("/documents/{doc_id}/chunks", response_model=GenericResponse)
def api_get_document_chunks(doc_id: int):
    try:
        return GenericResponse(data=get_document_chunks(doc_id))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/import/folder", response_model=GenericResponse)
def api_import_folder(req: ImportFolderRequest):
    try:
        result = import_documents(req.folder)
        return GenericResponse(data=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/import/file", response_model=GenericResponse)
def api_import_file(req: ImportFileRequest):
    try:
        result = import_single_document(req.file_path)
        return GenericResponse(data=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/index", response_model=GenericResponse)
def api_index(req: IndexRequest):
    try:
        result = index_document(
            doc_id=req.doc_id,
            chunk_size=req.chunk_size or DEFAULT_CHUNK_SIZE,
            overlap=req.overlap or DEFAULT_CHUNK_OVERLAP,
        )
        return GenericResponse(data=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/ask", response_model=AskResponse)
def api_ask(req: AskRequest):
    try:
        result = answer_question(req.question, top_k=req.top_k)
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/summary", response_model=GenericResponse)
def api_summary(req: SummaryRequest):
    try:
        result = summarize_text(req.text)
        return GenericResponse(data={"summary": result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e