import hashlib
import inspect
import json
import re
from typing import Any

from app.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.db import (
    clear_chunks_by_document_id,
    ensure_chunk_search_indexes,
    get_document_by_id,
    insert_chunk,
)
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


SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[。！？!?；;])\s+|(?<=\.)\s+(?=[A-Z0-9])|(?<=\n)"
)
HEADING_CN_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百千万0-9]+[章节部分节]|[一二三四五六七八九十]+[、.．]|[0-9]+[.、])"
)
LIST_ITEM_PATTERN = re.compile(r"^([-*•]\s+|[0-9]+[.)、]\s+)")
WHITESPACE_RE = re.compile(r"\s+")
LEXICAL_CLEAN_RE = re.compile(r"[^\w\u4e00-\u9fff]+")


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
            "search_text",
            "lexical_text",
            "doc_title",
            "section_title",
            "token_count",
            "chunk_hash",
        ]
        args = [payload.get(k) for k in common_order if k in payload]
        return insert_chunk(*args)


def _maybe_delete_vector_index(document_id: int) -> None:
    """
    可选钩子：
    如果未来新增 app.services.vector_store_service.delete_document_embeddings，
    这里会自动调用；如果没有该模块，则静默跳过。
    """
    try:
        from app.services.vector_store_service import delete_document_embeddings
    except Exception:
        return

    try:
        delete_document_embeddings(document_id)
    except Exception:
        # 不阻塞主索引流程
        return


def _maybe_upsert_vector_index(
    *,
    chunk_id: int | None,
    document_id: int,
    chunk_index: int,
    embedding: list[float] | None,
    chunk: dict[str, Any],
) -> None:
    """
    可选钩子：
    如果未来新增 app.services.vector_store_service.upsert_chunk_embedding，
    这里会自动调用；如果没有该模块，则静默跳过。
    """
    if not embedding:
        return

    try:
        from app.services.vector_store_service import upsert_chunk_embedding
    except Exception:
        return

    payload = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "chunk_index": chunk_index,
        "doc_title": chunk.get("doc_title"),
        "section_title": chunk.get("section_title"),
        "section_path": section_path_to_str(chunk.get("section_path")),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "chunk_type": chunk.get("chunk_type"),
        "token_count": chunk.get("token_count", 0),
        "chunk_hash": chunk.get("chunk_hash"),
        "search_text": chunk.get("search_text", ""),
        "chunk_text": chunk.get("chunk_text", ""),
        "metadata": chunk.get("metadata") or {},
    }

    try:
        upsert_chunk_embedding(
            chunk_id=chunk_id,
            embedding=embedding,
            payload=payload,
        )
    except Exception:
        # 不阻塞主索引流程
        return


def _estimate_token_count(text: str) -> int:
    text = text or ""
    if not text:
        return 0

    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_words = len(re.findall(r"[A-Za-z0-9_./:-]+", text))

    # 粗略估算：中文1字≈1 token，英文1词≈1.3 token
    return int(cjk_chars + ascii_words * 1.3)


def _hash_text(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def _normalize_lexical_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = text.lower()
    text = LEXICAL_CLEAN_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_whitespace(str(x)) for x in value if normalize_whitespace(str(x))]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [normalize_whitespace(str(x)) for x in parsed if normalize_whitespace(str(x))]
        except Exception:
            pass
        return [normalize_whitespace(value)]
    return [normalize_whitespace(str(value))]


def _coalesce_page_range(
    page_start: int | None,
    page_end: int | None,
    block: dict[str, Any],
) -> tuple[int | None, int | None]:
    bps = to_int(block.get("page_start"))
    bpe = to_int(block.get("page_end"))

    if page_start is None:
        page_start = bps
    if bpe is not None:
        page_end = bpe
    elif page_end is None:
        page_end = bps

    return page_start, page_end


def block_to_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        metadata = block.get("metadata") or {}
        page_start = to_int(block.get("page_start"))
        page_end = to_int(block.get("page_end"))

        if page_start is None:
            page_start = to_int(block.get("page"))
        if page_end is None:
            page_end = page_start

        if page_start is None:
            page_start = to_int(metadata.get("page"))
        if page_end is None:
            page_end = to_int(metadata.get("page_end")) or page_start

        return {
            "type": normalize_whitespace(str(block.get("type", "paragraph") or "paragraph")).lower(),
            "text": normalize_whitespace(block.get("text", "")),
            "section_path": normalize_section_path(block.get("section_path")),
            "page_start": page_start,
            "page_end": page_end,
            "metadata": metadata,
        }

    metadata = getattr(block, "metadata", {}) or {}
    page_start = to_int(getattr(block, "page_start", None))
    page_end = to_int(getattr(block, "page_end", None))

    if page_start is None:
        page_start = to_int(getattr(block, "page", None))
    if page_end is None:
        page_end = page_start

    if page_start is None:
        page_start = to_int(metadata.get("page"))
    if page_end is None:
        page_end = to_int(metadata.get("page_end")) or page_start

    return {
        "type": normalize_whitespace(str(getattr(block, "type", "paragraph") or "paragraph")).lower(),
        "text": normalize_whitespace(getattr(block, "text", "")),
        "section_path": normalize_section_path(getattr(block, "section_path", [])),
        "page_start": page_start,
        "page_end": page_end,
        "metadata": metadata,
    }


def _guess_heading_level(line: str) -> int:
    line = (line or "").strip()
    if not line:
        return 1
    if re.match(r"^#{1,6}\s+", line):
        return min(6, len(line.split(" ")[0]))
    if re.match(r"^第[一二三四五六七八九十百千万0-9]+章", line):
        return 1
    if re.match(r"^第[一二三四五六七八九十百千万0-9]+节", line):
        return 2
    if re.match(r"^[一二三四五六七八九十]+[、.．]", line):
        return 2
    if re.match(r"^[0-9]+[.、]", line):
        dots = line.split(" ")[0].count(".")
        return min(4, max(2, dots + 2))
    return 1


def build_blocks_from_content(content: str) -> list[dict[str, Any]]:
    content = (content or "").strip()
    if not content:
        return []

    lines = [line.rstrip() for line in content.splitlines()]
    blocks: list[dict[str, Any]] = []
    current_para: list[str] = []
    section_path: list[str] = []

    def flush_paragraph() -> None:
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
            level = _guess_heading_level(line)
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

        if HEADING_CN_PATTERN.match(line):
            flush_paragraph()
            level = _guess_heading_level(line)
            section_path = section_path[: max(0, level - 1)]
            section_path.append(line)
            blocks.append(
                {
                    "type": "heading",
                    "text": line,
                    "section_path": list(section_path),
                    "page_start": None,
                    "page_end": None,
                    "metadata": {"level": level},
                }
            )
            continue

        if LIST_ITEM_PATTERN.match(line):
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


def _can_merge_blocks(prev: dict[str, Any], curr: dict[str, Any], min_len: int) -> bool:
    prev_type = prev.get("type")
    curr_type = curr.get("type")
    prev_text = prev.get("text", "")
    curr_text = curr.get("text", "")
    same_section = normalize_section_path(prev.get("section_path")) == normalize_section_path(curr.get("section_path"))

    if not same_section:
        return False

    if prev_type in {"heading", "table", "code"} or curr_type in {"heading", "table", "code"}:
        return False

    if prev_type == "list_item" and curr_type == "list_item":
        return True

    if prev_type in {"paragraph", "list_item"} and curr_type in {"paragraph", "list_item"}:
        if len(prev_text) < min_len:
            return True
        if len(curr_text) < max(20, min_len // 2):
            return True
        if prev_text.endswith(("：", ":", "；", ";", "，", ",")):
            return True

    return False


def _merge_short_adjacent_blocks(blocks: list[dict[str, Any]], min_len: int = 80) -> list[dict[str, Any]]:
    if not blocks:
        return []

    merged: list[dict[str, Any]] = []

    for raw_block in blocks:
        block = block_to_dict(raw_block)
        text = block.get("text", "")
        if not text:
            continue

        if not merged:
            merged.append(block)
            continue

        prev = merged[-1]
        if _can_merge_blocks(prev, block, min_len=min_len):
            prev["text"] = normalize_whitespace(prev.get("text", "") + "\n" + text)
            prev_start = prev.get("page_start")
            prev_end = prev.get("page_end")
            curr_start = block.get("page_start")
            curr_end = block.get("page_end")

            if prev_start is None:
                prev["page_start"] = curr_start
            if curr_end is not None:
                prev["page_end"] = curr_end
            elif prev_end is None:
                prev["page_end"] = curr_start
        else:
            merged.append(block)

    return merged


def _split_paragraphs(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"\n{2,}", text)
    return [normalize_whitespace(p) for p in parts if normalize_whitespace(p)]


def _split_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    parts = re.split(SENTENCE_BOUNDARY_PATTERN, text)
    parts = [normalize_whitespace(p) for p in parts if normalize_whitespace(p)]
    return parts or [text]


def _best_cut_position(text: str, chunk_size: int) -> int:
    if len(text) <= chunk_size:
        return len(text)

    window = text[:chunk_size]
    preferred_marks = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", "，", ",", " "]

    best = -1
    for mark in preferred_marks:
        pos = window.rfind(mark)
        if pos > int(chunk_size * 0.55):
            best = pos + len(mark)
            break

    if best == -1:
        best = chunk_size

    return max(1, best)


def split_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text = normalize_whitespace((text or "").strip())
    if not text:
        return []

    if chunk_size <= 0 or len(text) <= chunk_size:
        return [text]

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    paragraphs = _split_paragraphs(text)
    if len(paragraphs) > 1:
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            candidate = para if not current else f"{current}\n{para}"
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(normalize_whitespace(current))
                current = ""

            if len(para) <= chunk_size:
                current = para
                continue

            chunks.extend(split_text(para, chunk_size=chunk_size, overlap=overlap))

        if current:
            chunks.append(normalize_whitespace(current))

        return [c for c in chunks if c]

    sentences = _split_sentences(text)
    if len(sentences) > 1:
        chunks: list[str] = []
        current = ""

        for sent in sentences:
            candidate = sent if not current else f"{current} {sent}"
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(normalize_whitespace(current))

            if len(sent) <= chunk_size:
                current = sent
            else:
                start = 0
                while start < len(sent):
                    remaining = sent[start:]
                    cut = _best_cut_position(remaining, chunk_size)
                    part = normalize_whitespace(remaining[:cut])
                    if part:
                        chunks.append(part)
                    if start + cut >= len(sent):
                        break
                    start = max(start + 1, start + cut - overlap)
                current = ""

        if current:
            chunks.append(normalize_whitespace(current))

        return [c for c in chunks if c]

    chunks = []
    start = 0
    while start < len(text):
        remaining = text[start:]
        cut = _best_cut_position(remaining, chunk_size)
        part = normalize_whitespace(remaining[:cut])
        if part:
            chunks.append(part)

        if start + cut >= len(text):
            break

        start = max(start + 1, start + cut - overlap)

    return [c for c in chunks if c]


def _build_chunk_prefix(section_path: list[str], heading_buffer: list[str]) -> str:
    parts: list[str] = []

    section_title = last_section_title(section_path)
    if section_title:
        parts.append(f"标题：{section_title}")

    if heading_buffer:
        unique_headings: list[str] = []
        seen = set()
        for h in heading_buffer:
            h = normalize_whitespace(h)
            if h and h not in seen:
                unique_headings.append(h)
                seen.add(h)
        if unique_headings:
            parts.append("小节：" + " / ".join(unique_headings[-2:]))

    return "\n".join(parts).strip()


def _build_search_text(doc_title: str, section_path: list[str], heading_buffer: list[str], body_text: str) -> str:
    parts: list[str] = []

    doc_title = normalize_whitespace(doc_title)
    if doc_title:
        parts.append(f"文档：{doc_title}")

    prefix = _build_chunk_prefix(section_path, heading_buffer)
    if prefix:
        parts.append(prefix)

    body_text = normalize_whitespace(body_text)
    if body_text:
        parts.append(body_text)

    return "\n".join(parts).strip()


def _build_lexical_text(doc_title: str, section_path: list[str], heading_buffer: list[str], body_text: str) -> str:
    raw = " ".join(
        [
            normalize_whitespace(doc_title),
            section_path_to_str(section_path),
            " ".join(heading_buffer[-3:]) if heading_buffer else "",
            normalize_whitespace(body_text),
        ]
    ).strip()
    return _normalize_lexical_text(raw)


def split_blocks_into_chunks(
    doc_title: str,
    blocks: list[dict[str, Any]],
    max_tokens: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    normalized_blocks = [block_to_dict(b) for b in blocks]
    normalized_blocks = [b for b in normalized_blocks if b.get("text")]
    normalized_blocks = _merge_short_adjacent_blocks(normalized_blocks)

    if not normalized_blocks:
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
    current_heading_buffer: list[str] = []

    def flush_current() -> None:
        nonlocal current_texts, current_types, current_section_path
        nonlocal current_page_start, current_page_end
        nonlocal current_block_start, current_block_end
        nonlocal current_heading_buffer

        body_text = normalize_whitespace("\n".join(current_texts))
        if not body_text:
            current_texts = []
            current_types = []
            current_section_path = []
            current_page_start = None
            current_page_end = None
            current_block_start = None
            current_block_end = None
            return

        chunk_type = "mixed"
        unique_types = {t for t in current_types if t}
        if len(unique_types) == 1:
            chunk_type = list(unique_types)[0]

        search_text = _build_search_text(
            doc_title=doc_title,
            section_path=current_section_path,
            heading_buffer=current_heading_buffer,
            body_text=body_text,
        )
        lexical_text = _build_lexical_text(
            doc_title=doc_title,
            section_path=current_section_path,
            heading_buffer=current_heading_buffer,
            body_text=body_text,
        )

        chunks.append(
            {
                "doc_title": doc_title,
                "chunk_text": body_text,
                "search_text": search_text,
                "lexical_text": lexical_text,
                "section_path": list(current_section_path),
                "section_title": last_section_title(current_section_path),
                "page_start": current_page_start,
                "page_end": current_page_end,
                "block_start_index": current_block_start,
                "block_end_index": current_block_end,
                "chunk_type": chunk_type,
                "token_count": _estimate_token_count(search_text),
                "chunk_hash": _hash_text(search_text),
                "metadata": {
                    "char_len": len(search_text),
                    "body_char_len": len(body_text),
                    "block_count": len(current_texts),
                    "source_block_types": list(sorted(unique_types)),
                    "has_list": "list_item" in unique_types,
                    "has_table": "table" in unique_types,
                    "heading_path_depth": len(current_section_path),
                    "heading_buffer": list(current_heading_buffer[-3:]),
                },
            }
        )

        current_texts = []
        current_types = []
        current_section_path = []
        current_page_start = None
        current_page_end = None
        current_block_start = None
        current_block_end = None

    for idx, block in enumerate(normalized_blocks):
        btype = (block.get("type") or "paragraph").lower()
        btext = normalize_whitespace(block.get("text", ""))
        if not btext:
            continue

        section_path = normalize_section_path(block.get("section_path"))
        metadata = block.get("metadata") or {}

        if btype == "heading":
            flush_current()
            current_heading_buffer.append(btext)

            level = to_int(metadata.get("level")) or len(section_path) or 1
            if len(current_heading_buffer) > 6:
                current_heading_buffer = current_heading_buffer[-6:]
            if level <= 1:
                current_heading_buffer = current_heading_buffer[-1:]
            continue

        estimated_search_text = _build_search_text(
            doc_title=doc_title,
            section_path=section_path,
            heading_buffer=current_heading_buffer,
            body_text=btext,
        )

        if len(estimated_search_text) > max_chars:
            flush_current()
            sub_chunks = split_text(estimated_search_text, chunk_size=max_chars, overlap=overlap_chars)

            for sub_idx, sub_text in enumerate(sub_chunks):
                chunks.append(
                    {
                        "doc_title": doc_title,
                        "chunk_text": btext,
                        "search_text": sub_text,
                        "lexical_text": _normalize_lexical_text(sub_text),
                        "section_path": list(section_path),
                        "section_title": last_section_title(section_path),
                        "page_start": to_int(block.get("page_start")),
                        "page_end": to_int(block.get("page_end")),
                        "block_start_index": idx,
                        "block_end_index": idx,
                        "chunk_type": btype,
                        "token_count": _estimate_token_count(sub_text),
                        "chunk_hash": _hash_text(sub_text),
                        "metadata": {
                            "sub_chunk_index": sub_idx,
                            "char_len": len(sub_text),
                            "body_char_len": len(btext),
                            "block_count": 1,
                            "source_block_types": [btype],
                            "has_list": btype == "list_item",
                            "has_table": btype == "table",
                            "heading_path_depth": len(section_path),
                            "heading_buffer": list(current_heading_buffer[-3:]),
                        },
                    }
                )
            continue

        current_prefix_text = _build_search_text(
            doc_title=doc_title,
            section_path=current_section_path if current_texts else section_path,
            heading_buffer=current_heading_buffer,
            body_text="\n".join(current_texts + [btext]),
        )
        projected_len = len(normalize_whitespace(current_prefix_text))

        should_flush = False
        if current_texts:
            if normalize_section_path(current_section_path) != section_path:
                should_flush = True
            elif projected_len > max_chars:
                should_flush = True

        if should_flush:
            flush_current()

        if not current_texts:
            current_section_path = list(section_path)
            current_page_start = to_int(block.get("page_start"))
            current_block_start = idx

        current_texts.append(btext)
        current_types.append(btype)
        current_page_start, current_page_end = _coalesce_page_range(
            current_page_start,
            current_page_end,
            block,
        )
        current_block_end = idx

    flush_current()
    return chunks


def _to_indexed_chunk_item(chunk_id: int | None, chunk_index: int, chunk: dict[str, Any]) -> dict[str, Any]:
    search_text = normalize_whitespace(chunk.get("search_text", ""))

    return {
        "id": chunk_id,
        "chunk_index": chunk_index,
        "chunk_type": chunk.get("chunk_type"),
        "section_path": section_path_to_str(chunk.get("section_path")),
        "section_title": chunk.get("section_title"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "token_count": chunk.get("token_count", 0),
        "block_start_index": chunk.get("block_start_index"),
        "block_end_index": chunk.get("block_end_index"),
        "preview": search_text[:200],
    }


def index_document(
    document_id: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    row = get_document_by_id(document_id)
    if not row:
        raise ValueError("document not found")

    title = safe_get(row, "title", "") or ""
    content = safe_get(row, "content", "") or ""
    blocks = safe_json_loads(safe_get(row, "blocks_json"), default=[])

    if not blocks:
        blocks = build_blocks_from_content(content)

    # 确保检索侧全文索引已存在
    ensure_chunk_search_indexes()

    # 重建前同时清理数据库 chunk 与可选向量索引
    clear_chunks_by_document_id(document_id)
    _maybe_delete_vector_index(document_id)

    chunks = split_blocks_into_chunks(
        doc_title=title,
        blocks=blocks,
        max_tokens=chunk_size,
        overlap=overlap,
    )

    saved_chunks: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        search_text = normalize_whitespace(chunk.get("search_text", ""))
        chunk_text = normalize_whitespace(chunk.get("chunk_text", ""))
        lexical_text = normalize_whitespace(chunk.get("lexical_text", ""))

        if not search_text:
            continue

        embedding = get_embedding(search_text)
        metadata = {
            "doc_title": chunk.get("doc_title"),
            "section_title": chunk.get("section_title"),
            **(chunk.get("metadata") or {}),
        }

        payload = {
            "document_id": document_id,
            "chunk_text": chunk_text,
            "search_text": search_text,
            "lexical_text": lexical_text,
            "embedding": embedding,
            "chunk_index": idx,
            "section_path": chunk.get("section_path") or [],
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "block_start_index": chunk.get("block_start_index"),
            "block_end_index": chunk.get("block_end_index"),
            "chunk_type": chunk.get("chunk_type"),
            "doc_title": chunk.get("doc_title"),
            "section_title": chunk.get("section_title"),
            "token_count": chunk.get("token_count", 0),
            "chunk_hash": chunk.get("chunk_hash"),
            "metadata_json": metadata,
        }

        inserted = _call_insert_chunk(payload)
        chunk_id = inserted if isinstance(inserted, int) else (
            inserted.get("id") if isinstance(inserted, dict) else None
        )

        _maybe_upsert_vector_index(
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=idx,
            embedding=embedding,
            chunk=chunk,
        )

        saved_chunks.append(_to_indexed_chunk_item(chunk_id=chunk_id, chunk_index=idx, chunk=chunk))

    return {
        "document_id": document_id,
        "title": title,
        "chunk_count": len(saved_chunks),
        "chunks": saved_chunks,
    }