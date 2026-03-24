from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


TOOL_VERSION = "v1"


class ToolError(BaseModel):
    code: str
    message: str


class ToolMeta(BaseModel):
    tool_name: str
    version: str = TOOL_VERSION
    duration_ms: Optional[int] = None


class ToolResult(BaseModel):
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[ToolError] = None
    meta: ToolMeta


# -------------------------
# Shared payload models
# -------------------------

class DocumentBrief(BaseModel):
    document_id: int
    title: Optional[str] = None
    status: Optional[str] = None
    message: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: Optional[str] = None


# -------------------------
# Import tool schemas
# -------------------------

class KBImportFileInput(BaseModel):
    file_path: str = Field(..., description="Absolute or relative path to a local file.")


class KBImportFileOutput(BaseModel):
    document_id: int
    title: Optional[str] = None
    status: Literal["imported", "failed", "skipped"] = "imported"
    message: str = "document imported successfully"


class KBImportFolderInput(BaseModel):
    folder: str = Field(..., description="Absolute or relative path to a local folder.")


class KBImportFolderOutput(BaseModel):
    count: int
    documents: List[DocumentBrief]


# -------------------------
# Index tool schemas
# -------------------------

class KBIndexDocumentInput(BaseModel):
    document_id: int = Field(..., ge=1)
    chunk_size: int = Field(800, ge=100, le=4000)
    overlap: int = Field(120, ge=0, le=1000)


class KBIndexDocumentOutput(BaseModel):
    document_id: int
    chunk_count: Optional[int] = None
    vector_count: Optional[int] = None
    status: str = "indexed"
    raw: Dict[str, Any] = Field(default_factory=dict)


# -------------------------
# Summary tool schemas
# -------------------------

class KBSummarizeDocumentInput(BaseModel):
    document_id: int = Field(..., ge=1)


class KBSummarizeDocumentOutput(BaseModel):
    document_id: int
    summary: str


# -------------------------
# History / session schemas
# -------------------------

class KBCreateChatSessionInput(BaseModel):
    session_id: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KBCreateChatSessionOutput(BaseModel):
    session_id: str
    title: Optional[str] = None
    summary_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class KBGetChatHistoryInput(BaseModel):
    session_id: str
    limit: int = Field(20, ge=1, le=100)


class KBGetChatHistoryOutput(BaseModel):
    session_id: str
    title: Optional[str] = None
    summary_text: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)

# -------------------------
# Search tool schemas
# -------------------------

class SearchHit(BaseModel):
    chunk_id: Optional[int] = None
    document_id: Optional[int] = None
    title: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_type: Optional[str] = None
    section_path: Optional[str] = None
    section_title: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    score: Optional[float] = None
    text: str = ""
    preview: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KBSearchKnowledgeBaseInput(BaseModel):
    query: str = Field(..., min_length=1, description="Search query for the knowledge base.")
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of hits to return.")
    include_full_text: bool = Field(
        default=True,
        description="Whether to include full chunk text in each hit.",
    )
    text_max_length: int = Field(
        default=2000,
        ge=100,
        le=10000,
        description="Maximum text length per hit when include_full_text is false.",
    )


class KBSearchKnowledgeBaseOutput(BaseModel):
    query: str
    top_k: int
    count: int
    hits: List[SearchHit] = Field(default_factory=list)