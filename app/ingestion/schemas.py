from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any


class DocumentBlock(BaseModel):
    block_id: str
    block_type: Literal["title", "heading", "paragraph", "table", "caption", "list"]
    text: str
    order: int
    page_num: Optional[int] = None
    level: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    title: str
    source_type: Literal["upload", "folder", "url"] = "folder"
    file_type: Literal["txt", "md", "pdf", "docx"]
    source_path: Optional[str] = None

    raw_text: str = ""
    clean_text: str = ""

    blocks: List[DocumentBlock] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)