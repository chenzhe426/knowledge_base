from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.ingestion.normalizers import clean_blocks, normalize_text
from app.ingestion.parsers.base import BaseParser
from app.ingestion.schemas import ParsedDocument


def _section_path_push(section_stack: list[str], heading_text: str, level: int) -> str:
    while len(section_stack) >= level:
        section_stack.pop()
    section_stack.append(heading_text)
    return " > ".join(section_stack)


def _docling_markdown_to_blocks(markdown_text: str) -> list[dict[str, Any]]:
    lines = normalize_text(markdown_text).split("\n")
    blocks: list[dict[str, Any]] = []
    section_stack: list[str] = []
    para_buf: list[str] = []

    def flush_paragraph():
        nonlocal para_buf
        if para_buf:
            text = "\n".join(para_buf).strip()
            if text:
                blocks.append(
                    {
                        "type": "paragraph",
                        "text": text,
                        "section_path": " > ".join(section_stack) if section_stack else None,
                    }
                )
        para_buf = []

    for line in lines:
        s = line.strip()
        if not s:
            flush_paragraph()
            continue

        if s.startswith("#"):
            flush_paragraph()
            level = len(s) - len(s.lstrip("#"))
            heading_text = s[level:].strip()
            section_path = _section_path_push(section_stack, heading_text, level)
            blocks.append(
                {
                    "type": "heading",
                    "text": heading_text,
                    "heading_level": level,
                    "section_path": section_path,
                }
            )
            continue

        if re.match(r"^\s*([-*•]|\d+[.)、])\s+.+$", s):
            flush_paragraph()
            blocks.append(
                {
                    "type": "list_item",
                    "text": s,
                    "section_path": " > ".join(section_stack) if section_stack else None,
                }
            )
            continue

        if s.startswith("|") and s.endswith("|"):
            flush_paragraph()
            blocks.append(
                {
                    "type": "table",
                    "text": s,
                    "section_path": " > ".join(section_stack) if section_stack else None,
                }
            )
            continue

        para_buf.append(s)

    flush_paragraph()
    return clean_blocks(blocks)


def _parse_with_docling(path: Path) -> list[dict[str, Any]]:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    markdown_text = result.document.export_to_markdown()
    return _docling_markdown_to_blocks(markdown_text)


def _parse_with_unstructured(path: Path) -> list[dict[str, Any]]:
    from unstructured.partition.auto import partition

    elements = partition(filename=str(path))
    blocks: list[dict[str, Any]] = []
    section_stack: list[str] = []

    for idx, el in enumerate(elements):
        text = getattr(el, "text", "") or ""
        text = text.strip()
        if not text:
            continue

        el_type = el.__class__.__name__
        meta = getattr(el, "metadata", None)

        if el_type == "Title":
            heading_text = text
            level = 1
            while len(section_stack) >= level:
                section_stack.pop()
            section_stack.append(heading_text)

            blocks.append(
                {
                    "type": "heading",
                    "text": heading_text,
                    "heading_level": level,
                    "section_path": " > ".join(section_stack),
                    "page": getattr(meta, "page_number", None) if meta else None,
                    "block_index": idx,
                }
            )
        else:
            block_type = {
                "NarrativeText": "paragraph",
                "ListItem": "list_item",
                "Table": "table",
                "Image": "image_caption",
            }.get(el_type, "paragraph")

            blocks.append(
                {
                    "type": block_type,
                    "text": text,
                    "section_path": " > ".join(section_stack) if section_stack else None,
                    "page": getattr(meta, "page_number", None) if meta else None,
                    "block_index": idx,
                }
            )

    return clean_blocks(blocks)


def _parse_with_python_docx(path: Path) -> list[dict[str, Any]]:
    from docx import Document

    doc = Document(str(path))
    blocks: list[dict[str, Any]] = []
    section_stack: list[str] = []

    for i, para in enumerate(doc.paragraphs):
        text = (para.text or "").strip()
        if not text:
            continue

        style_name = ""
        try:
            style_name = para.style.name or ""
        except Exception:
            style_name = ""

        style_lower = style_name.lower()

        if style_lower.startswith("heading"):
            m = re.search(r"heading\s*(\d+)", style_lower)
            level = int(m.group(1)) if m else 1
            section_path = _section_path_push(section_stack, text, level)
            blocks.append(
                {
                    "type": "heading",
                    "text": text,
                    "heading_level": level,
                    "section_path": section_path,
                    "block_index": i,
                }
            )
            continue

        if re.match(r"^\s*([-*•]|\d+[.)、])\s+.+$", text):
            blocks.append(
                {
                    "type": "list_item",
                    "text": text,
                    "section_path": " > ".join(section_stack) if section_stack else None,
                    "block_index": i,
                }
            )
            continue

        blocks.append(
            {
                "type": "paragraph",
                "text": text,
                "section_path": " > ".join(section_stack) if section_stack else None,
                "block_index": i,
            }
        )

    return clean_blocks(blocks)


class DocxParser(BaseParser):
    file_type = "docx"

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        errors: list[str] = []

        for parser_name, parser_func in [
            ("docling", _parse_with_docling),
            ("unstructured", _parse_with_unstructured),
            ("python-docx", _parse_with_python_docx),
        ]:
            try:
                blocks = parser_func(path)
                if blocks:
                    content = "\n\n".join(
                        b["text"] for b in blocks if b.get("type") != "heading"
                    ).strip()
                    return ParsedDocument(
                        title=path.stem,
                        content=content,
                        blocks=blocks,
                        source_path=str(path),
                        file_type=self.file_type,
                        metadata={"parser_used": parser_name, "errors": errors},
                    )
            except Exception as e:
                errors.append(f"{parser_name}: {e}")

        raise RuntimeError(f"failed to parse docx: {path} | errors={errors}")