from pydantic import BaseModel
from typing import Optional


class DocumentResponse(BaseModel):
    id: int
    title: str
    content: Optional[str] = None
    file_path: Optional[str] = None
    created_at: Optional[str] = None