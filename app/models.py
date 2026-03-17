from typing import Optional

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    id: int
    title: str
    content: Optional[str] = None
    file_path: Optional[str] = None
    created_at: Optional[str] = None


class ImportRequest(BaseModel):
    folder: str


class ImportResponse(BaseModel):
    imported_count: int
    document_ids: list[int]


class IndexRequest(BaseModel):
    chunk_size: int = Field(default=500, ge=100, le=3000)
    overlap: int = Field(default=100, ge=0, le=1000)


class ChunkResponse(BaseModel):
    chunk_id: Optional[int] = None
    document_id: int
    title: str
    chunk_index: int
    text: str
    score: Optional[float] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class IndexResponse(BaseModel):
    document_id: int
    title: str
    chunk_count: int
    chunks: list[dict]


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)


class RetrieveResponse(BaseModel):
    query: str
    results: list[ChunkResponse]


class AskRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[ChunkResponse]