from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ImportFolderRequest(BaseModel):
    folder: str


class ImportFileRequest(BaseModel):
    file_path: str


class IndexRequest(BaseModel):
    doc_id: int
    chunk_size: int | None = None
    overlap: int | None = None


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class SummaryRequest(BaseModel):
    text: str


class DocumentItem(BaseModel):
    id: int
    title: str
    file_path: str | None = None
    file_type: str | None = None
    source_type: str | None = None
    char_count: int = 0
    block_count: int = 0
    created_at: str | None = None


class ChunkItem(BaseModel):
    id: int
    document_id: int | None = None
    chunk_type: str | None = None
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    token_count: int | None = None
    preview: str


class SourceItem(BaseModel):
    document_id: int
    chunk_id: int
    score: float
    chunk_type: str | None = None
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    preview: str


class GenericResponse(BaseModel):
    success: bool = True
    data: Any


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]