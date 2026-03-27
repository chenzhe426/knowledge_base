import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from app.services.vector_store import vector_store

from app.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.db import (
    clear_chunks_by_document_id,
    ensure_chunk_search_indexes,
    get_document_by_id,
    insert_chunks_batch,
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

# Table-aware patterns
TABLE_ROW_SEP = re.compile(r"\n")
TABLE_CELL_SEP = re.compile(r"\s*\|\s*")
TABLE_NUMERIC_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\.\d+%?")
TABLE_PERCENT_RE = re.compile(r"\d+(\.\d+)?%")
TABLE_YEAR_RE = re.compile(r"(?<![.\d])(19|20)\d{2}(?![.\d])")
TABLE_UNIT_RE = re.compile(
    r"(?:单位|币种|货币|单位：|币种：|货币：)\s*([^\n\|]+)",
    re.IGNORECASE,
)
TABLE_TITLE_RE = re.compile(
    r"(?:表格|表[题名]?|数据来源|source)\s*[:：]?\s*([^\n\|]{2,50})",
    re.IGNORECASE,
)




def _maybe_delete_vector_index(document_id: int) -> None:
    if not document_id:
        return

    try:
        from app.services.vector_store import vector_store
        vector_store.delete_document_chunks(int(document_id))
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


# ---------------------------------------------------------------------------
# Table-aware chunking
# ---------------------------------------------------------------------------

def _parse_table_rows(table_text: str) -> list[list[str]]:
    """Parse table text into a list of rows, each row a list of cell strings.

    Table text format (from pdfplumber/docx): "A | B | C\\nD | E | F"
    Returns [[A, B, C], [D, E, F]]
    """
    lines = TABLE_ROW_SEP.split(table_text.strip())
    rows: list[list[str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in TABLE_CELL_SEP.split(line)]
        # Filter out empty cells at the end that came from trailing separators
        while cells and not cells[-1]:
            cells.pop()
        if cells:
            rows.append(cells)
    return rows


def _extract_table_metadata(
    rows: list[list[str]],
    table_text: str,
    section_path: list[str],
) -> dict[str, Any]:
    """Extract structured metadata from a parsed table.

    Heuristics:
    - First row with fewer cells than the second is treated as a header.
    - Table title is looked for near the top of the text.
    - Units / currency are extracted from header row or preamble.
    - Year and numeric signals scanned across all cells.
    """
    metadata: dict[str, Any] = {
        "table_title": "",
        "table_headers": [],
        "table_units": "",
        "row_start": 0,
        "row_end": len(rows) - 1,
        "contains_numeric_values": False,
        "has_percent": False,
        "has_year": False,
        "col_count": 0,
    }

    if not rows:
        return metadata

    # --- Table title: look for "表:" or "表格:" patterns near top ---
    for line in table_text.split("\n")[:5]:
        m = TABLE_TITLE_RE.search(line)
        if m:
            metadata["table_title"] = normalize_whitespace(m.group(1))
            break

    # --- Determine header row: first row that differs in pattern from data rows ---
    # Heuristic for financial tables:
    #   - First row with fewer columns → header (multi-row header)
    #   - First row's non-first-column cells are years ("2021") or period labels
    #     while second row's non-first-column cells are decimal numbers → first row is header
    if len(rows) >= 2:
        row0 = rows[0]
        row1 = rows[1]
        row0_cells = len(row0)
        row1_cells = len(row1)

        # Case 1: row 0 has fewer columns (e.g., merged header cells)
        if row0_cells < row1_cells and row0_cells >= 1:
            metadata["table_headers"] = row0
        else:
            # Case 2: header row has year/period labels in non-first columns,
            # data row has decimal numbers. Use heuristic:
            # - row0 non-first cells look like years ("2021") or short labels
            # - row1 non-first cells look like decimal numbers (100.5, 20.5%)
            r0_non_first = [c for c in row0[1:] if c]
            r1_non_first = [c for c in row1[1:] if c]

            r0_looks_like_header = (
                r0_non_first
                and all(
                    TABLE_YEAR_RE.match(c.strip()) or len(c.strip()) <= 5
                    for c in r0_non_first
                )
            )
            r1_looks_like_data = r1_non_first and any(
                "." in c or "%" in c or TABLE_NUMERIC_RE.match(c.strip())
                for c in r1_non_first
            )
            if r0_looks_like_header and r1_looks_like_data:
                metadata["table_headers"] = row0
    elif len(rows) == 1:
        metadata["table_headers"] = rows[0]

    metadata["col_count"] = max((len(r) for r in rows), default=0)

    # --- Units / currency: scan header cells and table preamble ---
    unit_candidates: set[str] = set()
    for cell in (metadata["table_headers"] or []):
        m = TABLE_UNIT_RE.search(cell)
        if m:
            unit_candidates.add(normalize_whitespace(m.group(1)))

    pre_lines = table_text.split("\n")[:3]
    for line in pre_lines:
        m = TABLE_UNIT_RE.search(line)
        if m:
            unit_candidates.add(normalize_whitespace(m.group(1)))

    if unit_candidates:
        metadata["table_units"] = " | ".join(sorted(unit_candidates))

    # --- Numeric signals: scan all data cells ---
    flat_text = " ".join(" ".join(r) for r in rows)
    if TABLE_NUMERIC_RE.search(flat_text):
        metadata["contains_numeric_values"] = True
    if TABLE_PERCENT_RE.search(flat_text):
        metadata["has_percent"] = True
    if TABLE_YEAR_RE.search(flat_text):
        metadata["has_year"] = True

    return metadata


def _split_table_by_rows(
    title: str,
    section_path: list[str],
    heading_buffer: list[str],
    table_text: str,
    table_meta: dict[str, Any],
    max_chars: int,
    overlap_chars: int,
    block_idx: int,
    page_start: int | None,
    page_end: int | None,
) -> list[dict[str, Any]]:
    """Split a table into row-level chunks, each chunk containing:
    - headers (repeated)
    - a window of data rows
    - structured metadata
    """
    rows = _parse_table_rows(table_text)
    if not rows:
        return []

    header_row = table_meta.get("table_headers") or []
    col_count = table_meta.get("col_count", 0)
    table_title = table_meta.get("table_title") or ""
    table_units = table_meta.get("table_units") or ""
    has_year = table_meta.get("has_year", False)

    chunks: list[dict[str, Any]] = []

    # Build header section once
    header_line = " | ".join(header_row) if header_row else ""
    meta_parts: list[str] = []
    if table_title:
        meta_parts.append(f"表：{table_title}")
    if table_units:
        meta_parts.append(f"单位：{table_units}")
    if has_year:
        # Extract year range from table using finditer to get full matches
        unique_years: list[str] = []
        seen_years: set[str] = set()
        for m in TABLE_YEAR_RE.finditer(table_text):
            full_year = m.group()  # e.g. "2021"
            if full_year not in seen_years:
                seen_years.add(full_year)
                unique_years.append(full_year)
                if len(unique_years) >= 5:
                    break
        if unique_years:
            meta_parts.append(f"年份：{', '.join(unique_years)}")

    header_block = "\n".join(meta_parts)

    # --- Determine data row range (skip header row for data) ---
    data_start_idx = 0
    if header_row and rows and rows[0] == header_row:
        data_start_idx = 1

    data_rows = rows[data_start_idx:]
    if not data_rows:
        # Table has no data rows, just header — emit as single chunk
        body_lines = [header_line] if header_line else []
        body_text = "\n".join(body_lines)
        chunk = _make_table_chunk(
            title=title,
            section_path=section_path,
            heading_buffer=heading_buffer,
            body_text=body_text,
            header_block=header_block,
            row_start=0,
            row_end=0,
            table_meta=table_meta,
            block_idx=block_idx,
            page_start=page_start,
            page_end=page_end,
        )
        return [chunk]

    # --- Sliding window over data rows, targeting max_chars ---
    current_lines: list[str] = []
    current_start_row = data_start_idx

    def table_chunk_size(lines: list[str]) -> int:
        """Estimate total search_text length for a table chunk."""
        body = "\n".join(lines)
        prefix = _build_table_search_text(
            title=title,
            section_path=section_path,
            heading_buffer=heading_buffer,
            header_block=header_block,
            body_rows=lines,
        )
        return len(prefix) + len(body)

    for i, row in enumerate(data_rows):
        row_line = " | ".join(row)
        projected = table_chunk_size(current_lines + [row_line])

        if projected > max_chars and current_lines:
            # Emit current chunk
            row_end = current_start_row + len(current_lines) - 1
            chunk = _make_table_chunk(
                title=title,
                section_path=section_path,
                heading_buffer=heading_buffer,
                body_text="\n".join(current_lines),
                header_block=header_block,
                row_start=current_start_row,
                row_end=row_end,
                table_meta=table_meta,
                block_idx=block_idx,
                page_start=page_start,
                page_end=page_end,
            )
            chunks.append(chunk)

            # Start next window with overlap
            overlap_rows = overlap_chars // (col_count * 4) if col_count > 0 else 1
            overlap_rows = max(1, min(overlap_rows, 3))
            overlap_start = max(current_start_row + len(current_lines) - overlap_rows, data_start_idx)
            current_lines = [" | ".join(r) for r in rows[overlap_start:i]]
            current_start_row = overlap_start
        else:
            if not current_lines:
                current_start_row = data_start_idx + i
            current_lines.append(row_line)

    # Emit final chunk
    if current_lines:
        row_end = current_start_row + len(current_lines) - 1
        chunk = _make_table_chunk(
            title=title,
            section_path=section_path,
            heading_buffer=heading_buffer,
            body_text="\n".join(current_lines),
            header_block=header_block,
            row_start=current_start_row,
            row_end=row_end,
            table_meta=table_meta,
            block_idx=block_idx,
            page_start=page_start,
            page_end=page_end,
        )
        chunks.append(chunk)

    return chunks


def _make_table_chunk(
    title: str,
    section_path: list[str],
    heading_buffer: list[str],
    body_text: str,
    header_block: str,
    row_start: int,
    row_end: int,
    table_meta: dict[str, Any],
    block_idx: int,
    page_start: int | None,
    page_end: int | None,
) -> dict[str, Any]:
    """Construct a single table chunk dict."""
    search_text = _build_table_search_text(
        title=title,
        section_path=section_path,
        heading_buffer=heading_buffer,
        header_block=header_block,
        body_rows=body_text.split("\n") if body_text else [],
    )
    lexical_text = _build_table_lexical_text(
        title=title,
        section_path=section_path,
        heading_buffer=heading_buffer,
        header_block=header_block,
        body_text=body_text,
    )

    return {
        "title": title,
        "chunk_text": body_text,
        "search_text": search_text,
        "lexical_text": lexical_text,
        "section_path": list(section_path),
        "section_title": last_section_title(section_path),
        "page_start": page_start,
        "page_end": page_end,
        "block_start_index": block_idx,
        "block_end_index": block_idx,
        "chunk_type": "table",
        "token_count": _estimate_token_count(search_text),
        "chunk_hash": _hash_text(search_text),
        "metadata": {
            "char_len": len(search_text),
            "body_char_len": len(body_text),
            "block_count": 1,
            "source_block_types": ["table"],
            "has_table": True,
            # Table-specific metadata
            "table_title": table_meta.get("table_title", ""),
            "table_headers": table_meta.get("table_headers", []),
            "table_units": table_meta.get("table_units", ""),
            "row_start": row_start,
            "row_end": row_end,
            "contains_numeric_values": table_meta.get("contains_numeric_values", False),
            "has_percent": table_meta.get("has_percent", False),
            "has_year": table_meta.get("has_year", False),
            "col_count": table_meta.get("col_count", 0),
            "heading_path_depth": len(section_path),
            "heading_buffer": list(heading_buffer[-3:]) if heading_buffer else [],
        },
    }


def _build_table_search_text(
    title: str,
    section_path: list[str],
    heading_buffer: list[str],
    header_block: str,
    body_rows: list[str],
) -> str:
    """Build search_text for a table chunk.

    Structure:
    文档：{title}
    标题：{section_title}
    小节：{heading_buffer}
    {header_block}
    表头：{header_line}
    ---数据行---
    {row_1}
    {row_2}
    ...
    """
    parts: list[str] = []

    title = normalize_whitespace(title)
    if title:
        parts.append(f"文档：{title}")

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

    if header_block:
        parts.append(header_block)

    if body_rows:
        parts.append("---数据行---")
        for row in body_rows:
            row = normalize_whitespace(row)
            if row:
                parts.append(row)

    return "\n".join(parts).strip()


def _build_table_lexical_text(
    title: str,
    section_path: list[str],
    heading_buffer: list[str],
    header_block: str,
    body_text: str,
) -> str:
    """Build lexical_text for a table chunk.

    Unlike regular lexical_text, preserve table structure markers (|, ---)
    but normalize whitespace and lowercase.
    """
    parts = [
        normalize_whitespace(title),
        section_path_to_str(section_path),
    ]
    if heading_buffer:
        parts.extend(normalize_whitespace(h) for h in heading_buffer[-3:] if h)
    if header_block:
        parts.append(normalize_whitespace(header_block))

    # Body: keep structure but normalize
    lines = [normalize_whitespace(l) for l in (body_text or "").splitlines() if l.strip()]
    for line in lines:
        parts.append(line)

    raw = " ".join(parts).strip()
    return _normalize_lexical_text(raw)


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


def _build_search_text(title: str, section_path: list[str], heading_buffer: list[str], body_text: str) -> str:
    parts: list[str] = []

    title = normalize_whitespace(title)
    if title:
        parts.append(f"文档：{title}")

    prefix = _build_chunk_prefix(section_path, heading_buffer)
    if prefix:
        parts.append(prefix)

    body_text = normalize_whitespace(body_text)
    if body_text:
        parts.append(body_text)

    return "\n".join(parts).strip()


def _build_lexical_text(title: str, section_path: list[str], heading_buffer: list[str], body_text: str) -> str:
    raw = " ".join(
        [
            normalize_whitespace(title),
            section_path_to_str(section_path),
            " ".join(heading_buffer[-3:]) if heading_buffer else "",
            normalize_whitespace(body_text),
        ]
    ).strip()
    return _normalize_lexical_text(raw)


def split_blocks_into_chunks(
    title: str,
    blocks: list[dict[str, Any]],
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    normalized_blocks = [block_to_dict(b) for b in blocks]
    normalized_blocks = [b for b in normalized_blocks if b.get("text")]
    normalized_blocks = _merge_short_adjacent_blocks(normalized_blocks)

    if not normalized_blocks:
        return []

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
            title=title,
            section_path=current_section_path,
            heading_buffer=current_heading_buffer,
            body_text=body_text,
        )
        lexical_text = _build_lexical_text(
            title=title,
            section_path=current_section_path,
            heading_buffer=current_heading_buffer,
            body_text=body_text,
        )

        chunks.append(
            {
                "title": title,
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

        # ── Table blocks: always treat as standalone, never accumulate ─────
        if btype == "table":
            flush_current()
            parsed_rows = _parse_table_rows(btext)
            table_meta = _extract_table_metadata(parsed_rows, btext, section_path)
            table_chunks = _split_table_by_rows(
                title=title,
                section_path=section_path,
                heading_buffer=current_heading_buffer,
                table_text=btext,
                table_meta=table_meta,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                block_idx=idx,
                page_start=to_int(block.get("page_start")),
                page_end=to_int(block.get("page_end")),
            )
            chunks.extend(table_chunks)
            continue

        estimated_search_text = _build_search_text(
            title=title,
            section_path=section_path,
            heading_buffer=current_heading_buffer,
            body_text=btext,
        )

        if len(estimated_search_text) > max_chars:
            flush_current()

            # ── Table blocks are handled above (line 1057); this branch is for oversized non-table blocks ──
            # ── Generic hard-split for non-table blocks ────────────────────
            sub_chunks = split_text(estimated_search_text, chunk_size=max_chars, overlap=overlap_chars)

            for sub_idx, sub_text in enumerate(sub_chunks):
                chunks.append(
                    {
                        "title": title,
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
            title=title,
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
        title=title,
        blocks=blocks,
        max_chars=chunk_size,
        overlap=overlap,
    )

    # 1. 预处理：收集所有有效 search_text
    valid_chunks: list[tuple[int, dict[str, Any]]] = []
    for idx, chunk in enumerate(chunks):
        search_text = normalize_whitespace(chunk.get("search_text", ""))
        if not search_text:
            continue
        valid_chunks.append((idx, chunk))

    # 2. 收集所有 search_text，为批量 embedding 准备
    search_texts = [normalize_whitespace(c[1].get("search_text", "")) for c in valid_chunks]

    # 3. 嵌入与插入流水线：后台线程计算 embedding，主线程写入 DB
    #    避免顺序等待，最大化重叠 embedding（I/O）与 DB 写入（I/O）
    vector_payloads: list[dict[str, Any]] = []
    saved_chunks: list[dict[str, Any]] = []

    def embed_worker(idx: int, chunk: dict[str, Any], search_text: str) -> tuple[int, dict[str, Any], list[float] | None]:
        embedding = get_embedding(search_text)
        return idx, chunk, embedding

    # 3. 并行 embedding，收集所有结果
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(embed_worker, idx, chunk, st): idx
            for (idx, chunk), st in zip(valid_chunks, search_texts)
        }

        results: list[tuple[int, dict[str, Any], list[float]]] = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                continue

    # 4. 构建批量插入 payload（不含 chunk_id）
    batch_payloads: list[dict[str, Any]] = []
    for idx, chunk, embedding in results:
        if not embedding:
            continue
        chunk_text = normalize_whitespace(chunk.get("chunk_text", ""))
        metadata = {
            "title": chunk.get("title"),
            "section_title": chunk.get("section_title"),
            **(chunk.get("metadata") or {}),
        }
        batch_payloads.append({
            "document_id": document_id,
            "chunk_text": chunk_text,
            "search_text": chunk.get("search_text", ""),
            "lexical_text": normalize_whitespace(chunk.get("lexical_text", "")),
            "embedding": embedding,
            "chunk_index": idx,
            "section_path": chunk.get("section_path") or [],
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "block_start_index": chunk.get("block_start_index"),
            "block_end_index": chunk.get("block_end_index"),
            "chunk_type": chunk.get("chunk_type"),
            "title": chunk.get("title"),
            "section_title": chunk.get("section_title"),
            "token_count": chunk.get("token_count", 0),
            "chunk_hash": chunk.get("chunk_hash"),
            "metadata_json": metadata,
        })

    # 5. 批量 MySQL 插入（一次 INSERT + 一次 SELECT + 一次 commit）
    if batch_payloads:
        chunk_ids = insert_chunks_batch(document_id, batch_payloads)
    else:
        chunk_ids = []

    # 6. 用返回的 chunk_id 构建 vector_payloads（保持顺序一致）
    for (idx, chunk, embedding), chunk_id in zip(results, chunk_ids):
        if not embedding or chunk_id is None or chunk_id == 0:
            saved_chunks.append(_to_indexed_chunk_item(chunk_id=None, chunk_index=idx, chunk=chunk))
            continue

        chunk_text = normalize_whitespace(chunk.get("chunk_text", ""))
        metadata = {
            "title": chunk.get("title"),
            "section_title": chunk.get("section_title"),
            **(chunk.get("metadata") or {}),
        }
        vector_payloads.append({
            "chunk_id": chunk_id,
            "document_id": document_id,
            "chunk_index": idx,
            "doc_title": chunk.get("title", ""),
            "section_title": chunk.get("section_title", ""),
            "section_path": section_path_to_str(chunk.get("section_path")),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "chunk_type": chunk.get("chunk_type"),
            "token_count": chunk.get("token_count", 0),
            "chunk_hash": chunk.get("chunk_hash"),
            "search_text": chunk.get("search_text", ""),
            "chunk_text": chunk_text,
            "metadata": metadata,
            "embedding": embedding,
        })
        saved_chunks.append(_to_indexed_chunk_item(chunk_id=chunk_id, chunk_index=idx, chunk=chunk))

    # 7. 批量 upsert 向量（一次调用）
    if vector_payloads:
        try:
            vector_store.upsert_chunks(vector_payloads)
        except Exception:
            pass

    return {
        "document_id": document_id,
        "title": title,
        "chunk_count": len(saved_chunks),
        "chunks": saved_chunks,
        "vector_count": len(saved_chunks),
        "status": "indexed",
    }