from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import requests

from app.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_MODEL,
)
from app.db import (
    clear_chunks_by_document_id,
    get_all_chunks,
    get_all_documents,
    get_chunks_by_document_id,
    get_document_by_id,
    insert_chunk,
    insert_document,
    search_documents,
)
from app.ingestion.pipeline import parse_document, parse_documents_from_folder
from app.ingestion.schemas import ParsedDocument


def _safe_json_loads(value: Any, default: Any):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


def _normalize_section_path(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if " > " in s:
            return [part.strip() for part in s.split(" > ") if part.strip()]
        return [s]
    return []


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_words = len([part for part in text.split() if part.strip()])
    punctuation_bonus = max(1, len(text) // 80)
    return chinese_chars + ascii_words + punctuation_bonus


def block_to_dict(block: Any, fallback_index: int = 0) -> dict[str, Any]:
    if hasattr(block, "model_dump"):
        block = block.model_dump()

    if not isinstance(block, dict):
        block = {
            "type": getattr(block, "type", "paragraph"),
            "text": getattr(block, "text", ""),
            "section_path": getattr(block, "section_path", None),
            "heading_level": getattr(block, "heading_level", None),
            "page": getattr(block, "page", None),
            "block_index": getattr(block, "block_index", fallback_index),
            "metadata": getattr(block, "metadata", {}) or {},
        }

    section_path = _normalize_section_path(block.get("section_path"))
    block_type = (block.get("type") or "paragraph").strip().lower()
    text = (block.get("text") or "").strip()

    return {
        "block_id": block.get("block_id") or f"block_{fallback_index}",
        "block_type": block_type,
        "text": text,
        "order": block.get("block_index", fallback_index),
        "page_num": block.get("page"),
        "level": block.get("heading_level"),
        "section_path": section_path,
        "metadata": block.get("metadata") or {},
    }


def normalize_block(block: dict[str, Any], fallback_order: int = 0) -> dict[str, Any]:
    text = (block.get("text") or "").strip()
    metadata = block.get("metadata") or {}

    return {
        "block_id": block.get("block_id") or f"block_{fallback_order}",
        "block_type": (block.get("block_type") or block.get("type") or "paragraph").strip().lower(),
        "text": text,
        "order": block.get("order", block.get("block_index", fallback_order)),
        "page_num": block.get("page_num", block.get("page")),
        "level": block.get("level", block.get("heading_level")),
        "section_path": _normalize_section_path(block.get("section_path")),
        "metadata": metadata,
    }


def parsed_document_to_db_payload(doc: ParsedDocument) -> dict[str, Any]:
    title = (doc.title or "").strip() or Path(doc.source_path or "").stem or "未命名文档"
    content = (doc.content or "").strip()

    blocks = [
        block_to_dict(block, fallback_index=idx)
        for idx, block in enumerate(doc.blocks or [])
    ]

    return {
        "title": title,
        "content": content,
        "raw_text": content,
        "file_path": doc.source_path,
        "file_type": doc.file_type,
        "source_type": "upload",
        "metadata": doc.metadata or {},
        "block_count": len(blocks),
        "blocks": blocks,
    }


def import_single_document(file_path: str) -> dict[str, Any]:
    parsed_doc = parse_document(file_path=file_path)
    payload = parsed_document_to_db_payload(parsed_doc)

    if not payload["content"]:
        raise ValueError(f"文档解析后内容为空: {file_path}")

    doc_id = insert_document(
        title=payload["title"],
        content=payload["content"],
        raw_text=payload["raw_text"],
        file_path=payload["file_path"],
        file_type=payload["file_type"],
        source_type=payload["source_type"],
        metadata=payload["metadata"],
        block_count=payload["block_count"],
        blocks=payload["blocks"],
    )

    return {
        "id": doc_id,
        "title": payload["title"],
        "file_path": payload["file_path"],
        "file_type": payload["file_type"],
        "source_type": payload["source_type"],
        "char_count": len(payload["content"]),
        "block_count": payload["block_count"],
        "metadata": payload["metadata"],
    }


def import_documents(folder: str) -> dict[str, Any]:
    parsed_documents = parse_documents_from_folder(folder_path=folder)
    imported: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for parsed_doc in parsed_documents:
        try:
            payload = parsed_document_to_db_payload(parsed_doc)
            if not payload["content"]:
                failed.append(
                    {
                        "file_path": payload["file_path"],
                        "reason": "empty content after parsing",
                    }
                )
                continue

            doc_id = insert_document(
                title=payload["title"],
                content=payload["content"],
                raw_text=payload["raw_text"],
                file_path=payload["file_path"],
                file_type=payload["file_type"],
                source_type=payload["source_type"],
                metadata=payload["metadata"],
                block_count=payload["block_count"],
                blocks=payload["blocks"],
            )

            imported.append(
                {
                    "id": doc_id,
                    "title": payload["title"],
                    "file_path": payload["file_path"],
                    "file_type": payload["file_type"],
                    "source_type": payload["source_type"],
                    "char_count": len(payload["content"]),
                    "block_count": payload["block_count"],
                    "metadata": payload["metadata"],
                }
            )
        except Exception as e:
            failed.append(
                {
                    "file_path": getattr(parsed_doc, "source_path", None),
                    "reason": str(e),
                }
            )

    return {
        "total": len(parsed_documents),
        "imported_count": len(imported),
        "failed_count": len(failed),
        "imported": imported,
        "failed": failed,
    }


def summarize_text(text: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一个擅长总结文档的助手，请用中文简洁总结。",
            },
            {
                "role": "user",
                "content": f"请总结下面内容，控制在 3 到 5 句话：\n\n{text}",
            },
        ],
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama 请求失败: {e}") from e
    except KeyError as e:
        raise RuntimeError("Ollama 返回格式异常") from e


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    chunks: list[dict[str, Any]] = []
    start = 0
    text = text.strip()
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "chunk_text": chunk_text,
                    "char_start": start,
                    "char_end": end,
                    "chunk_type": "paragraph",
                    "section_title": None,
                    "section_path": [],
                    "page_start": None,
                    "page_end": None,
                    "block_start_order": None,
                    "block_end_order": None,
                    "token_count": estimate_token_count(chunk_text),
                    "metadata": {"strategy": "legacy_char_split"},
                }
            )

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def build_blocks_from_content(content: str) -> list[dict[str, Any]]:
    paragraphs = [p.strip() for p in (content or "").split("\n\n") if p.strip()]
    blocks: list[dict[str, Any]] = []

    for idx, para in enumerate(paragraphs):
        blocks.append(
            {
                "block_id": f"fallback_{idx}",
                "block_type": "paragraph",
                "text": para,
                "order": idx,
                "page_num": None,
                "level": None,
                "section_path": [],
                "metadata": {},
            }
        )

    return blocks


def group_table_rows(table_text: str, max_chars: int) -> list[str]:
    text = table_text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    header = lines[0]
    rows = lines[1:]
    parts: list[str] = []
    current = header

    for row in rows:
        candidate = f"{current}\n{row}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            parts.append(current.strip())
            current = f"{header}\n{row}".strip()

    if current.strip():
        parts.append(current.strip())

    return parts


def build_chunk_text(
    doc_title: str,
    section_path: list[str],
    body_text: str,
    chunk_type: str,
) -> str:
    parts: list[str] = []

    if doc_title:
        parts.append(f"文档标题：{doc_title}")
    if section_path:
        parts.append(f"章节：{' > '.join(section_path)}")
    parts.append(f"内容类型：{chunk_type}")
    parts.append("")
    parts.append(body_text.strip())

    return "\n".join(parts).strip()


def finalize_chunk(
    doc_title: str,
    current_blocks: list[dict[str, Any]],
    section_path: list[str],
    chunk_type: str = "paragraph",
) -> dict[str, Any] | None:
    if not current_blocks:
        return None

    texts = [b["text"].strip() for b in current_blocks if b.get("text", "").strip()]
    if not texts:
        return None

    body_text = "\n\n".join(texts).strip()
    chunk_text = build_chunk_text(
        doc_title=doc_title,
        section_path=section_path,
        body_text=body_text,
        chunk_type=chunk_type,
    )

    pages = [b.get("page_num") for b in current_blocks if b.get("page_num") is not None]
    orders = [b.get("order") for b in current_blocks if b.get("order") is not None]

    return {
        "chunk_text": chunk_text,
        "chunk_type": chunk_type,
        "section_title": section_path[-1] if section_path else None,
        "section_path": list(section_path),
        "char_start": 0,
        "char_end": len(chunk_text),
        "page_start": min(pages) if pages else None,
        "page_end": max(pages) if pages else None,
        "block_start_order": min(orders) if orders else None,
        "block_end_order": max(orders) if orders else None,
        "token_count": estimate_token_count(chunk_text),
        "metadata": {
            "block_ids": [b["block_id"] for b in current_blocks],
            "block_types": [b["block_type"] for b in current_blocks],
            "strategy": "block_aware",
        },
    }


def split_blocks_into_chunks(
    *,
    doc_title: str,
    blocks: list[dict[str, Any]],
    max_tokens: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    if not blocks:
        return []

    normalized_blocks = [
        normalize_block(block, fallback_order=idx)
        for idx, block in enumerate(blocks)
        if (block.get("text") or "").strip()
    ]

    chunks: list[dict[str, Any]] = []
    current_blocks: list[dict[str, Any]] = []
    current_section: list[str] = []

    def flush_current():
        nonlocal current_blocks
        if not current_blocks:
            return
        chunk = finalize_chunk(
            doc_title=doc_title,
            current_blocks=current_blocks,
            section_path=current_section,
            chunk_type=current_blocks[0]["block_type"] if current_blocks else "paragraph",
        )
        if chunk:
            chunks.append(chunk)
        current_blocks = []

    for block in normalized_blocks:
        block_type = block["block_type"]
        text = block["text"]
        section_path = block["section_path"]
        token_count = estimate_token_count(text)

        if block_type == "heading":
            flush_current()
            current_section = section_path
            continue

        if block_type == "table":
            flush_current()
            table_parts = group_table_rows(text, max_chars=max(200, max_tokens * 2))
            for part_idx, part in enumerate(table_parts):
                single_block = [dict(block, text=part)]
                chunk = finalize_chunk(
                    doc_title=doc_title,
                    current_blocks=single_block,
                    section_path=section_path,
                    chunk_type="table",
                )
                if chunk:
                    chunk["metadata"]["table_part_index"] = part_idx
                    chunk["metadata"]["table_part_total"] = len(table_parts)
                    chunks.append(chunk)
            continue

        if not current_blocks:
            current_section = section_path

        current_text = "\n\n".join(b["text"] for b in current_blocks).strip()
        current_tokens = estimate_token_count(current_text)

        need_flush = False
        if current_blocks and section_path != current_section:
            need_flush = True
        elif current_blocks and current_tokens + token_count > max_tokens:
            need_flush = True

        if need_flush:
            flush_current()
            current_section = section_path

            if overlap > 0 and chunks:
                prev_blocks = current_blocks[-1:] if current_blocks else []
                current_blocks = [*prev_blocks] if prev_blocks else []

        current_blocks.append(block)

    flush_current()
    return chunks


def get_embedding(text: str) -> list[float]:
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "prompt": text,
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "embedding" not in data:
            raise RuntimeError("embedding 字段不存在")
        return data["embedding"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"embedding 请求失败: {e}") from e


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return -1.0

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return -1.0

    return dot / (norm1 * norm2)


def index_document(
    doc_id: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    row = get_document_by_id(doc_id)
    if not row:
        raise ValueError("document not found")

    title = row["title"]
    content = row.get("content") or ""
    blocks = row.get("blocks") or []

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
        embedding = get_embedding(chunk["chunk_text"])
        chunk_id = insert_chunk(
            document_id=doc_id,
            chunk_text=chunk["chunk_text"],
            embedding=embedding,
            chunk_type=chunk.get("chunk_type", "paragraph"),
            section_title=chunk.get("section_title"),
            section_path=chunk.get("section_path", []),
            char_start=chunk.get("char_start", 0),
            char_end=chunk.get("char_end", len(chunk["chunk_text"])),
            page_start=chunk.get("page_start"),
            page_end=chunk.get("page_end"),
            block_start_order=chunk.get("block_start_order"),
            block_end_order=chunk.get("block_end_order"),
            token_count=chunk.get("token_count", estimate_token_count(chunk["chunk_text"])),
            metadata=chunk.get("metadata", {}),
        )

        saved_chunks.append(
            {
                "id": chunk_id,
                "index": idx,
                "chunk_type": chunk.get("chunk_type"),
                "section_title": chunk.get("section_title"),
                "section_path": chunk.get("section_path", []),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "token_count": chunk.get("token_count"),
                "preview": chunk["chunk_text"][:160],
            }
        )

    return {
        "document_id": doc_id,
        "title": title,
        "chunk_count": len(saved_chunks),
        "chunks": saved_chunks,
    }


def retrieve_chunks(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    query_embedding = get_embedding(query)
    all_chunks = get_all_chunks()
    scored: list[dict[str, Any]] = []

    for chunk in all_chunks:
        emb = chunk.get("embedding")
        if not emb:
            continue
        score = cosine_similarity(query_embedding, emb)
        scored.append(
            {
                **chunk,
                "score": score,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def answer_question(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    retrieved = retrieve_chunks(query=query, top_k=top_k)
    context = "\n\n".join(item["chunk_text"] for item in retrieved)

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个基于知识库回答问题的助手。"
                    "请严格依据提供的上下文作答；"
                    "如果上下文不足，请明确说明“知识库中没有足够信息”。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{query}\n\n参考上下文：\n{context}",
            },
        ],
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer = data["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama 请求失败: {e}") from e
    except KeyError as e:
        raise RuntimeError("Ollama 返回格式异常") from e

    return {
        "question": query,
        "answer": answer,
        "sources": [
            {
                "document_id": item["document_id"],
                "chunk_id": item["id"],
                "score": item["score"],
                "chunk_type": item.get("chunk_type"),
                "section_title": item.get("section_title"),
                "section_path": item.get("section_path", []),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "preview": item["chunk_text"][:200],
            }
            for item in retrieved
        ],
    }


def list_documents() -> list[dict[str, Any]]:
    rows = get_all_documents()
    result = []
    for row in rows:
        result.append(
            {
                "id": row["id"],
                "title": row["title"],
                "file_path": row.get("file_path"),
                "file_type": row.get("file_type"),
                "source_type": row.get("source_type"),
                "char_count": len(row.get("content") or ""),
                "block_count": row.get("block_count", 0),
                "created_at": str(row.get("created_at")),
            }
        )
    return result


def get_document_chunks(doc_id: int) -> list[dict[str, Any]]:
    rows = get_chunks_by_document_id(doc_id)
    result = []
    for row in rows:
        result.append(
            {
                "id": row["id"],
                "document_id": row["document_id"],
                "chunk_type": row.get("chunk_type"),
                "section_title": row.get("section_title"),
                "section_path": row.get("section_path", []),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "token_count": row.get("token_count"),
                "preview": (row.get("chunk_text") or "")[:200],
            }
        )
    return result