from typing import Any

from pydantic import BaseModel, Field


class ImportRequest(BaseModel):
    folder: str = Field(..., description="要导入的文件夹路径")


class SingleFileImportRequest(BaseModel):
    file_path: str = Field(..., description="要导入的单个文件路径")
    source_type: str = Field(default="upload", description="来源类型，如 upload/folder/url")


class ImportedDocumentItem(BaseModel):
    id: int
    title: str
    file_path: str
    file_type: str | None = None
    source_type: str | None = None
    char_count: int
    block_count: int
    metadata: dict[str, Any] | None = None


class FailedImportItem(BaseModel):
    file_path: str | None = None
    reason: str


class ImportResponse(BaseModel):
    total: int
    imported_count: int
    failed_count: int
    imported: list[ImportedDocumentItem]
    failed: list[FailedImportItem]


class SingleFileImportResponse(BaseModel):
    id: int
    title: str
    file_path: str
    file_type: str | None = None
    source_type: str | None = None
    char_count: int
    block_count: int
    metadata: dict[str, Any] | None = None


class DocumentResponse(BaseModel):
    id: int
    title: str
    content: str | None = None
    file_path: str | None = None
    created_at: str | None = None


class IndexRequest(BaseModel):
    chunk_size: int = Field(default=500, ge=1, description="每个 chunk 的字符数")
    overlap: int = Field(default=100, ge=0, description="chunk 之间的重叠字符数")


class IndexedChunkItem(BaseModel):
    document_id: int
    title: str
    chunk_index: int
    char_start: int
    char_end: int
    text_preview: str


class IndexResponse(BaseModel):
    document_id: int
    title: str
    chunk_count: int
    chunks: list[IndexedChunkItem]


class ChunkResponse(BaseModel):
    chunk_id: int
    document_id: int
    title: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    score: float | None = None


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, description="召回片段数量")


class RetrieveResponse(BaseModel):
    query: str
    results: list[ChunkResponse]


class AskRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, description="召回片段数量")


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[ChunkResponse]