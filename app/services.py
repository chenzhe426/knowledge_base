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
    get_chunks_by_document_id,
    get_document_by_id,
    insert_chunk,
    insert_document,
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


def block_to_dict(block: Any, fallback_index: int = 0) -> dict[str, Any]:
    """
    把 ingestion 输出的 block 统一成 services 内部格式。
    新版 ingestion block 推荐字段：
    - type
    - text
    - section_path
    - heading_level
    - page
    - block_index
    """
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


def parsed_document_to_db_payload(doc: ParsedDocument) -> dict[str, Any]:
    """
    把 ParsedDocument 转成 documents 表可直接存储的 payload。
    新版 ParsedDocument:
    - title
    - content
    - blocks
    - source_path
    - file_type
    - metadata
    """
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
    """
    保留兼容版字符切分函数。
    正式索引流程优先走 block-aware chunking。
    """
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


def estimate_token_count(text: str) -> int:
    if not text:
        return 0

    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_words = len([part for part in text.split() if part.strip()])
    punctuation_bonus = max(1, len(text) // 80)
    return chinese_chars + ascii_words + punctuation_bonus


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


def build_blocks_from_content(content: str) -> list[dict[str, Any]]:
    """
    兼容旧数据：如果 documents 里没有 blocks_json，
    就把 content 按段落退化成 paragraph blocks。
    """
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
    """
    基于 blocks 的 chunk 切分。
    核心原则：
    1. heading 不单独落最终 chunk，主要用于更新 section_path
    2. paragraph / list_item 可聚合
    3. table 单独策略
    4. caption 尽量并入邻近正文
    """
    if not blocks:
        return []

    normalized_blocks = [
        normalize_block(block, fallback_order=idx)
        for idx, block in enumerate(blocks)
        if (block.get("text") or "").strip()
    ]
    if not normalized_blocks:
        return []

    chunks: list[dict[str, Any]] = []
    current_blocks: list[dict[str, Any]] = []
    current_tokens = 0
    current_section_path: list[str] = []

    def flush_current(chunk_type: str = "paragraph"):
        nonlocal current_blocks, current_tokens
        chunk = finalize_chunk(
            doc_title=doc_title,
            current_blocks=current_blocks,
            section_path=current_section_path,
            chunk_type=chunk_type,
        )
        if chunk:
            chunks.append(chunk)
        current_blocks = []
        current_tokens = 0

    def apply_heading(block: dict[str, Any]):
        nonlocal current_section_path

        if block.get("section_path"):
            current_section_path = list(block["section_path"])
            return

        heading_text = block["text"].strip()
        if not heading_text:
            return

        level = block.get("level")
        if level is None or level <= 0:
            level = 1

        while len(current_section_path) >= level:
            current_section_path.pop()

        current_section_path.append(heading_text)

    for block in normalized_blocks:
        block_type = block["block_type"]
        text = block["text"].strip()
        if not text:
            continue

        if block_type in {"heading", "title"}:
            flush_current("paragraph")
            apply_heading(block)
            continue

        if block_type == "table":
            flush_current("paragraph")
            table_parts = group_table_rows(text, max_chars=max(200, max_tokens))
            for part_idx, table_part in enumerate(table_parts):
                table_block = {
                    **block,
                    "text": table_part,
                    "metadata": {
                        **(block.get("metadata") or {}),
                        "table_part_index": part_idx,
                        "table_part_count": len(table_parts),
                    },
                }
                chunk = finalize_chunk(
                    doc_title=doc_title,
                    current_blocks=[table_block],
                    section_path=current_section_path,
                    chunk_type="table",
                )
                if chunk:
                    chunks.append(chunk)
            continue

        if block_type in {"image_caption", "caption"}:
            block_token = estimate_token_count(text)
            if current_blocks and current_tokens + block_token <= max_tokens:
                current_blocks.append(block)
                current_tokens += block_token
            else:
                chunk = finalize_chunk(
                    doc_title=doc_title,
                    current_blocks=[block],
                    section_path=current_section_path,
                    chunk_type="caption",
                )
                if chunk:
                    chunks.append(chunk)
            continue

        if block_type not in {"paragraph", "list_item", "quote", "code"}:
            block_type = "paragraph"

        block_token = estimate_token_count(text)

        if block_token > max_tokens:
            flush_current("paragraph")

            sentences = [s.strip() for s in text.split("\n") if s.strip()]
            if len(sentences) <= 1:
                hard_parts = [
                    text[i:i + max_tokens].strip()
                    for i in range(0, len(text), max_tokens)
                    if text[i:i + max_tokens].strip()
                ]
            else:
                hard_parts = []
                current_part = ""
                for sentence in sentences:
                    candidate = f"{current_part}\n{sentence}".strip() if current_part else sentence
                    if estimate_token_count(candidate) <= max_tokens:
                        current_part = candidate
                    else:
                        if current_part:
                            hard_parts.append(current_part)
                        current_part = sentence
                if current_part:
                    hard_parts.append(current_part)

            for part in hard_parts:
                single_block = {**block, "text": part}
                chunk = finalize_chunk(
                    doc_title=doc_title,
                    current_blocks=[single_block],
                    section_path=current_section_path,
                    chunk_type="list" if block["block_type"] == "list_item" else "paragraph",
                )
                if chunk:
                    chunks.append(chunk)
            continue

        if current_blocks and current_tokens + block_token > max_tokens:
            first_type = current_blocks[0]["block_type"]
            flush_current("list" if first_type == "list_item" else "paragraph")

        current_blocks.append(block)
        current_tokens += block_token

    if current_blocks:
        first_type = current_blocks[0]["block_type"]
        flush_current("list" if first_type == "list_item" else "paragraph")

    for chunk in chunks:
        chunk["metadata"]["overlap_mode"] = "semantic_heading_context"
        chunk["metadata"]["configured_overlap"] = overlap

    return chunks


def get_embedding(text: str) -> list[float]:
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "prompt": text,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError("Embedding 返回为空")
        return embedding
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Embedding 请求失败: {e}") from e


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


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
    blocks = _safe_json_loads(row.get("blocks_json"), default=[])

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

    for idx, item in enumerate(chunks):
        embedding = get_embedding(item["chunk_text"])
        insert_chunk(
            document_id=doc_id,
            chunk_index=idx,
            chunk_text=item["chunk_text"],
            embedding=embedding,
            char_start=item["char_start"],
            char_end=item["char_end"],
            chunk_type=item["chunk_type"],
            section_title=item["section_title"],
            section_path=item["section_path"],
            page_start=item["page_start"],
            page_end=item["page_end"],
            block_start_order=item["block_start_order"],
            block_end_order=item["block_end_order"],
            token_count=item["token_count"],
            metadata=item["metadata"],
        )
        saved_chunks.append(
            {
                "document_id": doc_id,
                "title": title,
                "chunk_index": idx,
                "chunk_type": item["chunk_type"],
                "section_title": item["section_title"],
                "section_path": item["section_path"],
                "page_start": item["page_start"],
                "page_end": item["page_end"],
                "block_start_order": item["block_start_order"],
                "block_end_order": item["block_end_order"],
                "token_count": item["token_count"],
                "text_preview": item["chunk_text"][:160],
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

    scored_results: list[dict[str, Any]] = []

    for row in all_chunks:
        try:
            chunk_embedding = json.loads(row["embedding"])
        except Exception:
            continue

        score = cosine_similarity(query_embedding, chunk_embedding)
        scored_results.append(
            {
                "chunk_id": row["id"],
                "document_id": row["document_id"],
                "title": row["title"],
                "chunk_index": row["chunk_index"],
                "chunk_type": row.get("chunk_type"),
                "section_title": row.get("section_title"),
                "section_path": _safe_json_loads(row.get("section_path_json"), default=[]),
                "text": row["chunk_text"],
                "score": round(score, 6),
                "char_start": row["char_start"],
                "char_end": row["char_end"],
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "block_start_order": row.get("block_start_order"),
                "block_end_order": row.get("block_end_order"),
                "token_count": row.get("token_count", 0),
                "metadata": _safe_json_loads(row.get("metadata_json"), default={}),
            }
        )

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:top_k]


def build_rag_prompt(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        section_text = " > ".join(chunk.get("section_path", []) or [])
        page_text = ""
        if chunk.get("page_start") is not None and chunk.get("page_end") is not None:
            page_text = f" | 页码: {chunk['page_start']}-{chunk['page_end']}"
        elif chunk.get("page_start") is not None:
            page_text = f" | 页码: {chunk['page_start']}"

        section_display = f" | 章节: {section_text}" if section_text else ""
        chunk_type = chunk.get("chunk_type") or "paragraph"

        context_parts.append(
            f"[来源{i}] 文档: {chunk['title']} | chunk: {chunk['chunk_index']} | 类型: {chunk_type}{section_display}{page_text}\n{chunk['text']}"
        )

    context_text = "\n\n".join(context_parts)

    return f"""
你是一个知识库问答助手，请严格根据提供的资料回答问题。

要求：
1. 只能依据“资料片段”作答，不要编造。
2. 如果资料不足以回答，请明确说“根据当前知识库内容无法确定”。
3. 回答使用中文。
4. 回答后附上你引用的来源编号，例如：[来源1]。
5. 尽量简洁清晰。
6. 优先利用章节、页码、表格等结构信息帮助定位答案。

用户问题：
{question}

资料片段：
{context_text}
""".strip()


def generate_answer(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    prompt = build_rag_prompt(question, retrieved_chunks)

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严谨的 RAG 问答助手。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"LLM 请求失败: {e}") from e
    except KeyError as e:
        raise RuntimeError("LLM 返回格式异常") from e


def answer_question(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    retrieved = retrieve_chunks(query=query, top_k=top_k)
    if not retrieved:
        return {
            "question": query,
            "answer": "当前知识库中没有可用的检索结果。",
            "sources": [],
        }

    answer = generate_answer(question=query, retrieved_chunks=retrieved)

    return {
        "question": query,
        "answer": answer,
        "sources": retrieved,
    }


def get_document_chunks(doc_id: int) -> list[dict[str, Any]]:
    rows = get_chunks_by_document_id(doc_id)

    results: list[dict[str, Any]] = []
    for row in rows:
        results.append(
            {
                "chunk_id": row["id"],
                "document_id": row["document_id"],
                "title": row["title"],
                "chunk_index": row["chunk_index"],
                "chunk_type": row.get("chunk_type"),
                "section_title": row.get("section_title"),
                "section_path": _safe_json_loads(row.get("section_path_json"), default=[]),
                "text": row["chunk_text"],
                "char_start": row["char_start"],
                "char_end": row["char_end"],
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "block_start_order": row.get("block_start_order"),
                "block_end_order": row.get("block_end_order"),
                "token_count": row.get("token_count", 0),
                "metadata": _safe_json_loads(row.get("metadata_json"), default={}),
            }
        )
    return results