import json
import math
import re
from pathlib import Path
from typing import Any

import requests

from app.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_NEAR_DUP_THRESHOLD,
    DEFAULT_NEIGHBOR_RADIUS,
    DEFAULT_PER_DOC_LIMIT,
    DEFAULT_RETRIEVE_CANDIDATES,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_TOP_K,
    HYBRID_WEIGHT_EMBEDDING,
    HYBRID_WEIGHT_LEXICAL,
    HYBRID_WEIGHT_METADATA,
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
)
from app.ingestion.pipeline import parse_document, parse_documents_from_folder
from app.ingestion.schemas import ParsedDocument


# =========================
# 基础工具
# =========================

_STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "是", "在", "对", "把", "被", "将", "并",
    "一个", "一种", "一些", "这个", "那个", "这些", "那些", "我们", "你们", "他们",
    "如何", "怎么", "怎样", "为什么", "是否", "什么", "哪些", "哪个", "哪里", "多少",
    "请问", "一下", "一下子", "一下儿", "一下吧", "一下呢", "一下吗",
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "what", "why", "how", "which", "where", "when", "is", "are", "was", "were",
}

QUESTION_HINTS = {
    "definition": ["什么是", "是什么", "定义", "含义", "概念"],
    "comparison": ["区别", "不同", "对比", "比较", "优缺点", "优点", "缺点"],
    "howto": ["如何", "怎么", "步骤", "实现", "做法", "流程"],
    "reason": ["为什么", "原因", "目的", "作用"],
    "location": ["哪里", "哪一章", "哪一节", "哪一页", "提到"],
}


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


def block_to_dict(block: Any) -> dict[str, Any]:
    if hasattr(block, "model_dump"):
        return block.model_dump()
    if isinstance(block, dict):
        return block
    return {
        "block_id": getattr(block, "block_id", ""),
        "block_type": getattr(block, "block_type", "paragraph"),
        "text": getattr(block, "text", ""),
        "order": getattr(block, "order", 0),
        "page_num": getattr(block, "page_num", None),
        "level": getattr(block, "level", None),
        "section_path": getattr(block, "section_path", []),
        "metadata": getattr(block, "metadata", {}) or {},
    }


def parsed_document_to_db_payload(doc: ParsedDocument) -> dict[str, Any]:
    """
    把 ParsedDocument 转成 documents 表可直接存储的 payload。
    """
    title = (doc.title or "").strip() or Path(doc.source_path or "").stem or "未命名文档"
    clean_text = (doc.clean_text or "").strip()
    raw_text = (doc.raw_text or "").strip()
    content = clean_text or raw_text
    blocks = [block_to_dict(block) for block in (doc.blocks or [])]

    return {
        "title": title,
        "content": content,
        "raw_text": raw_text or content,
        "file_path": doc.source_path,
        "file_type": doc.file_type,
        "source_type": doc.source_type,
        "metadata": doc.metadata or {},
        "block_count": len(blocks),
        "blocks": blocks,
    }


# =========================
# 导入
# =========================

def import_single_document(file_path: str, source_type: str = "upload") -> dict[str, Any]:
    parsed_doc = parse_document(file_path=file_path, source_type=source_type)
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
    parsed_documents = parse_documents_from_folder(folder_path=folder, recursive=True)

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


# =========================
# LLM / Embedding
# =========================

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


# =========================
# Chunk 基础
# =========================

def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    保留一个兼容版字符切分函数，避免其他地方仍有引用。
    但正式索引流程不再优先使用它。
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
    """
    粗略 token 估算：
    - 中文按字符近似
    - 英文按词近似
    """
    if not text:
        return 0

    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_words = len([part for part in text.split() if part.strip()])
    punctuation_bonus = max(1, len(text) // 80)
    return chinese_chars + ascii_words + punctuation_bonus


def _normalize_section_path(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return [str(x).strip() for x in loaded if str(x).strip()]
        except Exception:
            pass
        if ">" in text:
            return [part.strip() for part in text.split(">") if part.strip()]
        return [text]
    return []


def normalize_block(block: dict[str, Any], fallback_order: int = 0) -> dict[str, Any]:
    text = (block.get("text") or "").strip()
    metadata = block.get("metadata") or {}

    block_type = (block.get("block_type") or block.get("type") or "paragraph").strip().lower()
    if block_type in {"title", "header", "subtitle"}:
        block_type = "heading"
    elif block_type in {"list", "listitem", "bullet"}:
        block_type = "list_item"

    section_path = _normalize_section_path(
        block.get("section_path")
        or metadata.get("section_path")
        or metadata.get("section_titles")
    )

    order = block.get("order", block.get("block_index", fallback_order))
    page_num = block.get("page_num", block.get("page"))

    return {
        "block_id": block.get("block_id") or f"block_{fallback_order}",
        "block_type": block_type,
        "text": text,
        "order": order,
        "page_num": page_num,
        "level": block.get("level", block.get("heading_level")),
        "section_path": section_path,
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
    """
    表格文本拆分：
    - 短表直接一块
    - 长表按行拆
    """
    text = table_text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return [text[i: i + max_chars] for i in range(0, len(text), max_chars)]

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
            "strategy": "block_aware_v2",
        },
    }


_TRANSITION_PREFIXES = (
    "例如",
    "比如",
    "如下",
    "如下所示",
    "总结如下",
    "说明如下",
    "具体如下",
    "其特点如下",
    "其优点如下",
    "其缺点如下",
    "主要包括",
    "包括",
    "可分为",
    "分为",
    "如下图",
    "如下表",
    "见下文",
    "如下几点",
)


def is_transition_block(text: str) -> bool:
    s = (text or "").strip().rstrip("：:;；。")
    if not s:
        return False
    if len(s) > 24:
        return False
    return any(s.startswith(prefix) for prefix in _TRANSITION_PREFIXES)


def split_long_text_by_sentences(text: str, max_tokens: int) -> list[str]:
    """
    对超长 block 做句级拆分，尽量按中文/英文句边界切。
    """
    text = (text or "").strip()
    if not text:
        return []

    if estimate_token_count(text) <= max_tokens:
        return [text]

    pieces = re.split(r"(?<=[。！？!?；;])\s+|(?<=\.)\s+(?=[A-Z])", text)
    pieces = [p.strip() for p in pieces if p.strip()]

    if len(pieces) <= 1:
        pieces = re.split(r"\n+|(?<=[，,])", text)
        pieces = [p.strip() for p in pieces if p.strip()]

    if len(pieces) <= 1:
        approx_chars = max(150, max_tokens * 2)
        return [
            text[i: i + approx_chars].strip()
            for i in range(0, len(text), approx_chars)
            if text[i: i + approx_chars].strip()
        ]

    result: list[str] = []
    current = ""

    for piece in pieces:
        candidate = piece if not current else f"{current} {piece}"
        if estimate_token_count(candidate) <= max_tokens:
            current = candidate
        else:
            if current.strip():
                result.append(current.strip())
            if estimate_token_count(piece) <= max_tokens:
                current = piece
            else:
                approx_chars = max(150, max_tokens * 2)
                for i in range(0, len(piece), approx_chars):
                    sub = piece[i: i + approx_chars].strip()
                    if sub:
                        result.append(sub)
                current = ""

    if current.strip():
        result.append(current.strip())

    return result


def expand_blocks_for_chunking(
    blocks: list[dict[str, Any]],
    max_tokens: int,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block["block_type"]
        text = block["text"]

        if block_type in {"heading", "table"}:
            expanded.append(block)
            continue

        if estimate_token_count(text) <= max_tokens:
            expanded.append(block)
            continue

        parts = split_long_text_by_sentences(text, max_tokens=max(80, int(max_tokens * 0.75)))
        if len(parts) <= 1:
            expanded.append(block)
            continue

        total = len(parts)
        for idx, part in enumerate(parts):
            new_block = dict(block)
            new_block["text"] = part
            new_block["block_id"] = f'{block["block_id"]}_part_{idx}'
            new_block["metadata"] = {
                **(block.get("metadata") or {}),
                "split_from_block_id": block["block_id"],
                "split_part_index": idx,
                "split_part_total": total,
            }
            expanded.append(new_block)

    return expanded


def choose_overlap_blocks(blocks: list[dict[str, Any]], overlap: int) -> list[dict[str, Any]]:
    if overlap <= 0 or not blocks:
        return []

    selected: list[dict[str, Any]] = []
    total = 0

    for block in reversed(blocks):
        text = (block.get("text") or "").strip()
        if not text:
            continue

        tokens = estimate_token_count(text)

        if block.get("block_type") == "heading":
            continue

        if selected and total + tokens > overlap * 1.2:
            break

        selected.insert(0, block)
        total += tokens

        if total >= overlap:
            break

    return selected


def infer_chunk_type(blocks: list[dict[str, Any]]) -> str:
    if not blocks:
        return "paragraph"

    types = [b.get("block_type", "paragraph") for b in blocks]
    first = types[0]

    if first == "table":
        return "table"
    if first == "code":
        return "code"
    if all(t == "list_item" for t in types):
        return "list"
    if "list_item" in types:
        return "mixed"
    return first or "paragraph"


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
    if not normalized_blocks:
        return []

    normalized_blocks = expand_blocks_for_chunking(normalized_blocks, max_tokens=max_tokens)

    chunks: list[dict[str, Any]] = []
    current_blocks: list[dict[str, Any]] = []
    current_section: list[str] = []
    pending_transition_blocks: list[dict[str, Any]] = []

    def flush_current():
        nonlocal current_blocks
        if not current_blocks:
            return

        chunk = finalize_chunk(
            doc_title=doc_title,
            current_blocks=current_blocks,
            section_path=current_section,
            chunk_type=infer_chunk_type(current_blocks),
        )
        if chunk:
            chunk["metadata"]["has_transition_prefix"] = bool(
                current_blocks and is_transition_block(current_blocks[0].get("text", ""))
            )
            chunks.append(chunk)

        current_blocks = []

    def current_token_count() -> int:
        if not current_blocks:
            return 0
        return sum(estimate_token_count(b.get("text", "")) for b in current_blocks)

    for block in normalized_blocks:
        block_type = block["block_type"]
        text = block["text"].strip()
        section_path = block.get("section_path") or current_section
        token_count = estimate_token_count(text)

        if block_type == "heading":
            flush_current()
            current_section = section_path or [text]
            pending_transition_blocks = []
            continue

        if not current_section:
            current_section = section_path

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
            pending_transition_blocks = []
            continue

        if block_type in {"paragraph", "list_item"} and is_transition_block(text):
            if current_blocks:
                if current_token_count() + token_count <= max_tokens:
                    current_blocks.append(block)
                    continue
                flush_current()

            pending_transition_blocks.append(block)
            continue

        if current_blocks and section_path != current_section:
            flush_current()
            current_section = section_path

        if not current_blocks:
            current_section = section_path
            if pending_transition_blocks:
                candidate = pending_transition_blocks + [block]
                candidate_tokens = sum(estimate_token_count(b["text"]) for b in candidate)
                if candidate_tokens <= max_tokens:
                    current_blocks.extend(candidate)
                    pending_transition_blocks = []
                else:
                    tail = pending_transition_blocks[-1:]
                    current_blocks.extend(tail + [block])
                    pending_transition_blocks = []
            else:
                current_blocks.append(block)
            continue

        allow_soft_overrun = (
            current_blocks
            and current_blocks[-1].get("block_type") == "list_item"
            and block_type == "list_item"
        )

        projected = current_token_count() + token_count
        limit = int(max_tokens * 1.15) if allow_soft_overrun else max_tokens

        if projected > limit:
            prev_tail = choose_overlap_blocks(current_blocks, overlap=overlap)
            flush_current()
            current_blocks = list(prev_tail)

            current_section = section_path

            if pending_transition_blocks:
                candidate = current_blocks + pending_transition_blocks + [block]
                candidate_tokens = sum(estimate_token_count(b["text"]) for b in candidate)
                if candidate_tokens <= max_tokens:
                    current_blocks = candidate
                else:
                    current_blocks.extend(pending_transition_blocks[-1:] + [block])
                pending_transition_blocks = []
            else:
                current_blocks.append(block)
        else:
            if pending_transition_blocks:
                candidate_tokens = projected + sum(
                    estimate_token_count(b["text"]) for b in pending_transition_blocks
                )
                if candidate_tokens <= max_tokens:
                    current_blocks.extend(pending_transition_blocks)
                else:
                    current_blocks.extend(pending_transition_blocks[-1:])
                pending_transition_blocks = []

            current_blocks.append(block)

    if pending_transition_blocks:
        if current_blocks:
            current_blocks.extend(pending_transition_blocks)
        else:
            current_blocks = list(pending_transition_blocks)

    flush_current()
    return chunks


# =========================
# 索引
# =========================

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


# =========================
# 检索增强
# =========================

def normalize_text_for_search(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_query_terms(query: str) -> list[str]:
    text = normalize_text_for_search(query)
    if not text:
        return []

    raw_terms = re.findall(r"[\u4e00-\u9fff]{1,}|[a-zA-Z0-9_#+.\-]{2,}", text)
    terms: list[str] = []

    for term in raw_terms:
        t = term.strip().lower()
        if not t:
            continue
        if t in _STOPWORDS:
            continue
        if len(t) == 1 and not re.search(r"[\u4e00-\u9fff]", t):
            continue
        terms.append(t)

    unique_terms = []
    seen = set()
    for term in sorted(terms, key=len, reverse=True):
        if term not in seen:
            unique_terms.append(term)
            seen.add(term)

    return unique_terms


def infer_question_type(query: str) -> str:
    q = query.strip()
    for qtype, hints in QUESTION_HINTS.items():
        if any(h in q for h in hints):
            return qtype
    return "general"


def lexical_match_score(query_terms: list[str], text: str) -> float:
    if not query_terms:
        return 0.0

    norm_text = normalize_text_for_search(text)
    if not norm_text:
        return 0.0

    total = 0.0
    matched = 0

    for term in query_terms:
        if term in norm_text:
            matched += 1
            total += min(2.0, 0.6 + len(term) * 0.12)

    coverage = matched / max(1, len(query_terms))
    return min(1.0, (total / max(1, len(query_terms) * 1.6)) * 0.7 + coverage * 0.3)


def metadata_match_score(query_terms: list[str], chunk: dict[str, Any]) -> float:
    title = chunk.get("document_title") or chunk.get("title") or ""
    section_title = chunk.get("section_title") or ""
    section_path = " ".join(chunk.get("section_path") or [])
    chunk_type = chunk.get("chunk_type") or ""

    title_score = lexical_match_score(query_terms, title)
    section_score = lexical_match_score(query_terms, f"{section_title} {section_path}")

    bonus = 0.0
    if any(term in {"表", "表格", "table"} for term in query_terms) and chunk_type == "table":
        bonus += 0.2
    if any(term in {"代码", "code"} for term in query_terms) and chunk_type == "code":
        bonus += 0.2
    if any(term in {"定义", "概念", "是什么"} for term in query_terms):
        if "定义" in section_title or "概念" in section_title:
            bonus += 0.15

    return min(1.0, title_score * 0.55 + section_score * 0.45 + bonus)


def normalize_embedding_score(score: float) -> float:
    score = max(-1.0, min(1.0, score))
    return (score + 1) / 2


def compute_hybrid_score(
    query: str,
    query_terms: list[str],
    chunk: dict[str, Any],
    embedding_score: float,
) -> dict[str, float]:
    chunk_text = chunk.get("chunk_text") or ""
    lexical_score = lexical_match_score(query_terms, chunk_text)
    meta_score = metadata_match_score(query_terms, chunk)
    emb_score = normalize_embedding_score(embedding_score)

    qtype = infer_question_type(query)

    weight_embedding = HYBRID_WEIGHT_EMBEDDING
    weight_lexical = HYBRID_WEIGHT_LEXICAL
    weight_metadata = HYBRID_WEIGHT_METADATA

    if qtype == "definition":
        weight_embedding, weight_lexical, weight_metadata = 0.42, 0.28, 0.30
    elif qtype == "comparison":
        weight_embedding, weight_lexical, weight_metadata = 0.50, 0.30, 0.20
    elif qtype == "howto":
        weight_embedding, weight_lexical, weight_metadata = 0.52, 0.30, 0.18
    elif qtype == "location":
        weight_embedding, weight_lexical, weight_metadata = 0.32, 0.28, 0.40

    final_score = (
        emb_score * weight_embedding
        + lexical_score * weight_lexical
        + meta_score * weight_metadata
    )

    return {
        "embedding_score": round(emb_score, 6),
        "lexical_score": round(lexical_score, 6),
        "metadata_score": round(meta_score, 6),
        "final_score": round(final_score, 6),
    }


def chunk_signature_text(chunk: dict[str, Any]) -> str:
    text = chunk.get("chunk_text") or ""
    text = re.sub(r"文档标题：.*?\n", "", text)
    text = re.sub(r"章节：.*?\n", "", text)
    text = re.sub(r"内容类型：.*?\n", "", text)
    return normalize_text_for_search(text)


def text_jaccard_similarity(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def deduplicate_and_diversify(
    ranked_chunks: list[dict[str, Any]],
    top_k: int,
    per_doc_limit: int = DEFAULT_PER_DOC_LIMIT,
    near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    doc_counts: dict[int, int] = {}

    for chunk in ranked_chunks:
        doc_id = chunk.get("document_id")
        if doc_id is not None and doc_counts.get(doc_id, 0) >= per_doc_limit:
            continue

        signature = chunk_signature_text(chunk)
        is_dup = False
        for chosen in selected:
            sim = text_jaccard_similarity(signature, chunk_signature_text(chosen))
            same_doc = chosen.get("document_id") == doc_id
            same_section = (chosen.get("section_path") or []) == (chunk.get("section_path") or [])
            if sim >= near_dup_threshold or (same_doc and same_section and sim >= 0.65):
                is_dup = True
                break

        if is_dup:
            continue

        selected.append(chunk)
        if doc_id is not None:
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        if len(selected) >= top_k:
            break

    return selected


def build_chunk_neighbors(
    chunks: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    by_id: dict[int, dict[str, Any]] = {}
    by_doc: dict[int, list[dict[str, Any]]] = {}

    for chunk in chunks:
        chunk_id = chunk.get("id")
        doc_id = chunk.get("document_id")
        if chunk_id is not None:
            by_id[chunk_id] = chunk
        if doc_id is not None:
            by_doc.setdefault(doc_id, []).append(chunk)

    for doc_id, items in by_doc.items():
        items.sort(
            key=lambda x: (
                x.get("block_start_order") if x.get("block_start_order") is not None else 10**9,
                x.get("page_start") if x.get("page_start") is not None else 10**9,
                x.get("id") if x.get("id") is not None else 10**9,
            )
        )

    return by_id, by_doc


def get_adjacent_chunks(
    base_chunk: dict[str, Any],
    by_doc: dict[int, list[dict[str, Any]]],
    radius: int = DEFAULT_NEIGHBOR_RADIUS,
) -> list[dict[str, Any]]:
    doc_id = base_chunk.get("document_id")
    if doc_id is None or doc_id not in by_doc:
        return []

    items = by_doc[doc_id]
    target_id = base_chunk.get("id")
    idx = None
    for i, item in enumerate(items):
        if item.get("id") == target_id:
            idx = i
            break

    if idx is None:
        return []

    neighbors: list[dict[str, Any]] = []
    for offset in range(1, radius + 1):
        left = idx - offset
        right = idx + offset
        if left >= 0:
            neighbors.append(items[left])
        if right < len(items):
            neighbors.append(items[right])
    return neighbors


def rerank_chunks(
    query: str,
    query_terms: list[str],
    candidate_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reranked: list[dict[str, Any]] = []

    for chunk in candidate_chunks:
        emb_raw = chunk.get("_embedding_raw_score", chunk.get("score", -1.0))
        score_info = compute_hybrid_score(
            query=query,
            query_terms=query_terms,
            chunk=chunk,
            embedding_score=emb_raw,
        )
        reranked.append(
            {
                **chunk,
                "score": score_info["final_score"],
                "embedding_score": score_info["embedding_score"],
                "lexical_score": score_info["lexical_score"],
                "metadata_score": score_info["metadata_score"],
            }
        )

    reranked.sort(
        key=lambda x: (
            x["score"],
            x.get("metadata_score", 0),
            x.get("lexical_score", 0),
            x.get("embedding_score", 0),
        ),
        reverse=True,
    )
    return reranked


def retrieve_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    增强版检索：
    1. query embedding
    2. 对所有 chunks 做向量粗排
    3. 用关键词/标题/章节信息做规则重排
    4. 去重与多样性控制
    """
    query = (query or "").strip()
    if not query:
        return []

    query_terms = extract_query_terms(query)
    query_embedding = get_embedding(query)
    all_chunks = get_all_chunks()
    if not all_chunks:
        return []

    pre_candidates: list[dict[str, Any]] = []
    for chunk in all_chunks:
        emb = chunk.get("embedding")
        if not emb:
            continue

        raw_score = cosine_similarity(query_embedding, emb)
        pre_candidates.append(
            {
                **chunk,
                "_embedding_raw_score": raw_score,
                "score": raw_score,
            }
        )

    if not pre_candidates:
        return []

    pre_candidates.sort(key=lambda x: x["score"], reverse=True)

    coarse_k = min(
        max(top_k * 8, DEFAULT_RETRIEVE_CANDIDATES),
        len(pre_candidates),
    )
    coarse_candidates = pre_candidates[:coarse_k]

    doc_best_scores: dict[int, float] = {}
    for item in coarse_candidates:
        doc_id = item.get("document_id")
        if doc_id is None:
            continue
        doc_best_scores[doc_id] = max(
            doc_best_scores.get(doc_id, -1.0),
            item["_embedding_raw_score"],
        )

    top_doc_ids = {
        doc_id
        for doc_id, _ in sorted(
            doc_best_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[: max(3, top_k)]
    }

    stage2_candidates = [
        item for item in coarse_candidates
        if item.get("document_id") in top_doc_ids
        or item["_embedding_raw_score"] >= coarse_candidates[-1]["_embedding_raw_score"]
    ]

    rerank_limit = min(len(stage2_candidates), max(DEFAULT_RERANK_CANDIDATES, top_k * 4))
    reranked = rerank_chunks(
        query=query,
        query_terms=query_terms,
        candidate_chunks=stage2_candidates[:rerank_limit],
    )

    selected = deduplicate_and_diversify(
        ranked_chunks=reranked,
        top_k=top_k,
        per_doc_limit=DEFAULT_PER_DOC_LIMIT,
        near_dup_threshold=DEFAULT_NEAR_DUP_THRESHOLD,
    )
    return selected


def assemble_context(
    query: str,
    primary_chunks: list[dict[str, Any]],
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    neighbor_radius: int = DEFAULT_NEIGHBOR_RADIUS,
) -> tuple[str, list[dict[str, Any]]]:
    """
    组装最终上下文：
    - 先放主命中 chunks
    - 再适度补相邻 chunks
    - 控制上下文长度
    """
    if not primary_chunks:
        return "", []

    all_chunks = get_all_chunks()
    _, by_doc = build_chunk_neighbors(all_chunks)

    selected: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    total_tokens = 0

    def try_add(chunk: dict[str, Any], source_role: str):
        nonlocal total_tokens
        chunk_id = chunk.get("id")
        if chunk_id is None or chunk_id in seen_ids:
            return False

        chunk_tokens = chunk.get("token_count") or estimate_token_count(chunk.get("chunk_text", ""))
        if selected and total_tokens + chunk_tokens > max_context_tokens:
            return False

        item = dict(chunk)
        item["source_role"] = source_role
        selected.append(item)
        seen_ids.add(chunk_id)
        total_tokens += chunk_tokens
        return True

    for chunk in primary_chunks:
        try_add(chunk, "primary")

    for chunk in primary_chunks:
        neighbors = get_adjacent_chunks(chunk, by_doc=by_doc, radius=neighbor_radius)
        neighbors.sort(
            key=lambda x: (
                0 if (x.get("section_path") or []) == (chunk.get("section_path") or []) else 1,
                abs((x.get("block_start_order") or 0) - (chunk.get("block_start_order") or 0)),
            )
        )
        for nb in neighbors:
            if total_tokens >= max_context_tokens:
                break
            try_add(nb, "neighbor")

    context_parts = []
    for idx, item in enumerate(selected, start=1):
        context_parts.append(
            f"[片段{idx} | 来源={item.get('source_role')} | 分数={item.get('score', 0):.4f}]\n{item.get('chunk_text', '')}"
        )

    return "\n\n".join(context_parts), selected


# =========================
# 问答
# =========================

def answer_question(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    retrieved = retrieve_chunks(query=query, top_k=top_k)
    context, used_chunks = assemble_context(
        query=query,
        primary_chunks=retrieved,
        max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
        neighbor_radius=DEFAULT_NEIGHBOR_RADIUS,
    )

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个基于知识库回答问题的助手。"
                    "请严格依据提供的上下文作答；"
                    "优先给出准确、直接、可验证的回答；"
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
                "score": item.get("score"),
                "embedding_score": item.get("embedding_score"),
                "lexical_score": item.get("lexical_score"),
                "metadata_score": item.get("metadata_score"),
                "source_role": item.get("source_role", "primary"),
                "chunk_type": item.get("chunk_type"),
                "section_title": item.get("section_title"),
                "section_path": item.get("section_path", []),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "preview": (item.get("chunk_text") or "")[:220],
            }
            for item in used_chunks
        ],
        "retrieved_count": len(retrieved),
        "used_count": len(used_chunks),
    }


# =========================
# 查询接口
# =========================

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