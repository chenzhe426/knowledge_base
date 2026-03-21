import inspect
import json
import re
from typing import Any

from app.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.db import clear_chunks_by_document_id, get_document_by_id, insert_chunk
from app.services.common import (
    last_section_title,
    normalize_section_path,
    normalize_whitespace,
    safe_get,
    safe_json_loads,
    section_path_to_str,
    to_int,
)
from app.services.llm_service import get_embedding


def _call_insert_chunk(payload: dict[str, Any]) -> Any:
    sig = inspect.signature(insert_chunk)
    accepted = set(sig.parameters.keys())
    kwargs = {k: v for k, v in payload.items() if k in accepted}
    try:
        return insert_chunk(**kwargs)
    except TypeError:
        common_order = [
            "document_id",
            "chunk_text",
            "embedding",
            "chunk_index",
            "section_path",
            "page_start",
            "page_end",
            "block_start_index",
            "block_end_index",
            "chunk_type",
            "metadata_json",
        ]
        args = [payload.get(k) for k in common_order if k in payload]
        return insert_chunk(*args)


def block_to_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return {
            "type": block.get("type", "paragraph"),
            "text": normalize_whitespace(block.get("text", "")),
            "section_path": normalize_section_path(block.get("section_path")),
            "page_start": to_int(block.get("page_start")),
            "page_end": to_int(block.get("page_end")),
            "metadata": block.get("metadata") or {},
        }

    return {
        "type": getattr(block, "type", "paragraph"),
        "text": normalize_whitespace(getattr(block, "text", "")),
        "section_path": normalize_section_path(getattr(block, "section_path", [])),
        "page_start": to_int(getattr(block, "page_start", None)),
        "page_end": to_int(getattr(block, "page_end", None)),
        "metadata": getattr(block, "metadata", {}) or {},
    }


def build_blocks_from_content(content: str) -> list[dict[str, Any]]:
    content = (content or "").strip()
    if not content:
        return []

    lines = [line.rstrip() for line in content.splitlines()]
    blocks: list[dict[str, Any]] = []
    current_para: list[str] = []
    section_path: list[str] = []

    def flush_paragraph():
        nonlocal current_para
        text = normalize_whitespace("\n".join(current_para))
        if text:
            blocks.append(
                {
                    "type": "paragraph",
                    "text": text,
                    "section_path": list(section_path),
                    "page_start": None,
                    "page_end": None,
                    "metadata": {},
                }
            )
        current_para = []

    for raw in lines:
        line = raw.strip()
        if not line:
            flush_paragraph()
            continue

        if re.match(r"^#{1,6}\s+", line):
            flush_paragraph()
            level = len(line.split(" ")[0])
            title = re.sub(r"^#{1,6}\s+", "", line).strip()
            if title:
                section_path = section_path[: max(0, level - 1)]
                section_path.append(title)
                blocks.append(
                    {
                        "type": "heading",
                        "text": title,
                        "section_path": list(section_path),
                        "page_start": None,
                        "page_end": None,
                        "metadata": {"level": level},
                    }
                )
            continue

        if re.match(r"^(第[一二三四五六七八九十百千万0-9]+[章节部分节]|[一二三四五六七八九十]+[、.．]|[0-9]+[.、])", line):
            flush_paragraph()
            section_path = [line]
            blocks.append(
                {
                    "type": "heading",
                    "text": line,
                    "section_path": list(section_path),
                    "page_start": None,
                    "page_end": None,
                    "metadata": {"level": 1},
                }
            )
            continue

        if re.match(r"^[-*•]\s+", line) or re.match(r"^[0-9]+[.)、]\s+", line):
            flush_paragraph()
            blocks.append(
                {
                    "type": "list_item",
                    "text": line,
                    "section_path": list(section_path),
                    "page_start": None,
                    "page_end": None,
                    "metadata": {},
                }
            )
            continue

        current_para.append(line)

    flush_paragraph()
    return blocks


def _merge_short_adjacent_blocks(blocks: list[dict[str, Any]], min_len: int = 60) -> list[dict[str, Any]]:
    if not blocks:
        return []

    merged: list[dict[str, Any]] = []

    for block in blocks:
        block = block_to_dict(block)
        text = block.get("text", "")
        if not text:
            continue

        if not merged:
            merged.append(block)
            continue

        prev = merged[-1]
        can_merge = (
            prev.get("type") in {"paragraph", "list_item"}
            and block.get("type") in {"paragraph", "list_item"}
            and normalize_section_path(prev.get("section_path")) == normalize_section_path(block.get("section_path"))
            and len(prev.get("text", "")) < min_len
        )
        if can_merge:
            prev["text"] = normalize_whitespace(prev.get("text", "") + "\n" + text)
            prev["page_start"] = prev.get("page_start") if prev.get("page_start") is not None else block.get("page_start")
            prev["page_end"] = block.get("page_end") if block.get("page_end") is not None else prev.get("page_end")
        else:
            merged.append(block)

    return merged


def split_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    if chunk_size <= 0:
        return [text]

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def split_blocks_into_chunks(
    doc_title: str,
    blocks: list[dict[str, Any]],
    max_tokens: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    norm_blocks = _merge_short_adjacent_blocks([block_to_dict(b) for b in blocks if block_to_dict(b).get("text")])
    if not norm_blocks:
        return []

    max_chars = max_tokens
    overlap_chars = overlap

    chunks: list[dict[str, Any]] = []
    current_texts: list[str] = []
    current_types: list[str] = []
    current_section_path: list[str] = []
    current_page_start: int | None = None
    current_page_end: int | None = None
    current_block_start: int | None = None
    current_block_end: int | None = None

    def flush_current():
        nonlocal current_texts, current_types, current_section_path
        nonlocal current_page_start, current_page_end, current_block_start, current_block_end

        text = normalize_whitespace("\n".join(current_texts))
        if text:
            chunk_type = "mixed"
            unique_types = set(current_types)
            if len(unique_types) == 1:
                chunk_type = list(unique_types)[0]

            chunks.append(
                {
                    "doc_title": doc_title,
                    "chunk_text": text,
                    "section_path": list(current_section_path),
                    "section_title": last_section_title(current_section_path),
                    "page_start": current_page_start,
                    "page_end": current_page_end,
                    "block_start_index": current_block_start,
                    "block_end_index": current_block_end,
                    "chunk_type": chunk_type,
                    "metadata": {},
                }
            )

        current_texts = []
        current_types = []
        current_section_path = []
        current_page_start = None
        current_page_end = None
        current_block_start = None
        current_block_end = None

    for idx, block in enumerate(norm_blocks):
        btype = block.get("type", "paragraph")
        btext = normalize_whitespace(block.get("text", ""))
        if not btext:
            continue

        section_path = normalize_section_path(block.get("section_path"))
        page_start = to_int(block.get("page_start"))
        page_end = to_int(block.get("page_end"))

        if btype == "heading":
            flush_current()
            continue

        if len(btext) > max_chars:
            flush_current()
            sub_chunks = split_text(btext, chunk_size=max_chars, overlap=overlap_chars)
            for sub_idx, sub_text in enumerate(sub_chunks):
                chunks.append(
                    {
                        "doc_title": doc_title,
                        "chunk_text": sub_text,
                        "section_path": list(section_path),
                        "section_title": last_section_title(section_path),
                        "page_start": page_start,
                        "page_end": page_end,
                        "block_start_index": idx,
                        "block_end_index": idx,
                        "chunk_type": btype,
                        "metadata": {"sub_chunk_index": sub_idx},
                    }
                )
            continue

        current_len = len(normalize_whitespace("\n".join(current_texts)))
        should_flush = False

        if current_texts:
            if normalize_section_path(current_section_path) != section_path:
                should_flush = True
            elif current_len + 1 + len(btext) > max_chars:
                should_flush = True

        if should_flush:
            flush_current()

        if not current_texts:
            current_section_path = list(section_path)
            current_page_start = page_start
            current_block_start = idx

        current_texts.append(btext)
        current_types.append(btype)
        current_page_end = page_end if page_end is not None else current_page_end
        current_block_end = idx

    flush_current()
    return chunks


def index_document(
    doc_id: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    row = get_document_by_id(doc_id)
    if not row:
        raise ValueError("document not found")

    title = safe_get(row, "title", "") or ""
    content = safe_get(row, "content", "") or ""
    blocks = safe_json_loads(safe_get(row, "blocks_json"), default=[])

    if not blocks:
        blocks = build_blocks_from_content(content)

    clear_chunks_by_document_id(doc_id)

    chunks = split_blocks_into_chunks(
        doc_title=title,
        blocks=blocks,
        max_tokens=chunk_size,
        overlap=overlap,
    )

    saved_chunks: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        chunk_text = chunk.get("chunk_text", "").strip()
        if not chunk_text:
            continue

        embedding = get_embedding(chunk_text)
        payload = {
            "document_id": doc_id,
            "chunk_text": chunk_text,
            "embedding": json.dumps(embedding, ensure_ascii=False),
            "chunk_index": idx,
            "section_path": section_path_to_str(chunk.get("section_path")),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "block_start_index": chunk.get("block_start_index"),
            "block_end_index": chunk.get("block_end_index"),
            "chunk_type": chunk.get("chunk_type"),
            "metadata_json": json.dumps(
                {
                    "doc_title": chunk.get("doc_title"),
                    "section_title": chunk.get("section_title"),
                    **(chunk.get("metadata") or {}),
                },
                ensure_ascii=False,
            ),
        }

        inserted = _call_insert_chunk(payload)
        chunk_id = inserted if isinstance(inserted, int) else (
            inserted.get("id") if isinstance(inserted, dict) else None
        )

        saved_chunks.append(
            {
                "id": chunk_id,
                "chunk_index": idx,
                "chunk_type": chunk.get("chunk_type"),
                "section_path": section_path_to_str(chunk.get("section_path")),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "block_start_index": chunk.get("block_start_index"),
                "block_end_index": chunk.get("block_end_index"),
                "preview": normalize_whitespace(chunk_text)[:120],
            }
        )

    return {
        "document_id": doc_id,
        "title": title,
        "chunk_count": len(saved_chunks),
        "chunks": saved_chunks,
    }