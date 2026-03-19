from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.loaders import load_text_file
from app.ingestion.normalizers import normalize_text, clean_blocks
from app.ingestion.parsers.base import BaseParser
from app.ingestion.schemas import ParsedDocument


_HEADING_PATTERNS = [
    re.compile(r"^\s{0,3}#{1,6}\s+(.+)$"),
    re.compile(r"^\s*第[一二三四五六七八九十0-9]+[章节部分篇]\s+(.+)$"),
    re.compile(r"^\s*[一二三四五六七八九十]+[、.]\s*(.+)$"),
    re.compile(r"^\s*\d+(\.\d+){0,3}\s+(.+)$"),
    re.compile(r"^\s*[（(][一二三四五六七八九十0-9]+[)）]\s*(.+)$"),
]

_LIST_PATTERN = re.compile(r"^\s*([-*•]|\d+[.)、])\s+.+$")


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    for p in _HEADING_PATTERNS:
        if p.match(s):
            return True
    return False


def _is_list_item(line: str) -> bool:
    return bool(_LIST_PATTERN.match(line.strip()))


def _heading_level(text: str) -> int:
    s = text.strip()
    if s.startswith("#"):
        return len(s) - len(s.lstrip("#"))
    if re.match(r"^\s*\d+\.", s):
        return 2
    return 1


class TextParser(BaseParser):
    file_type = "text"

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        raw = load_text_file(path)
        text = normalize_text(raw)

        lines = [line.rstrip() for line in text.split("\n")]
        blocks: list[dict] = []

        section_stack: list[str] = []
        paragraph_buf: list[str] = []

        def flush_paragraph():
            nonlocal paragraph_buf, blocks
            if not paragraph_buf:
                return
            para = "\n".join(paragraph_buf).strip()
            if para:
                blocks.append(
                    {
                        "type": "paragraph",
                        "text": para,
                        "section_path": " > ".join(section_stack) if section_stack else None,
                    }
                )
            paragraph_buf = []

        for line in lines:
            s = line.strip()

            if not s:
                flush_paragraph()
                continue

            if _is_heading(s):
                flush_paragraph()
                level = _heading_level(s)
                heading_text = re.sub(r"^\s{0,3}#{1,6}\s+", "", s).strip()

                while len(section_stack) >= level:
                    section_stack.pop()
                section_stack.append(heading_text)

                blocks.append(
                    {
                        "type": "heading",
                        "text": heading_text,
                        "heading_level": level,
                        "section_path": " > ".join(section_stack),
                    }
                )
                continue

            if _is_list_item(s):
                flush_paragraph()
                blocks.append(
                    {
                        "type": "list_item",
                        "text": s,
                        "section_path": " > ".join(section_stack) if section_stack else None,
                    }
                )
                continue

            paragraph_buf.append(s)

        flush_paragraph()

        cleaned = clean_blocks(blocks)

        content = "\n\n".join(b["text"] for b in cleaned if b["type"] != "heading").strip()
        title = path.stem

        return ParsedDocument(
            title=title,
            content=content,
            blocks=cleaned,
            source_path=str(path),
            file_type=self.file_type,
            metadata={},
        )