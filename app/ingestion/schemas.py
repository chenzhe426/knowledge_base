from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedDocument:
    title: str
    content: str
    blocks: list[dict[str, Any]]
    source_path: str
    file_type: str
    metadata: dict[str, Any] = field(default_factory=dict)