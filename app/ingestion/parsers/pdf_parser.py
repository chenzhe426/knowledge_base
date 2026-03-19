from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber

from app.ingestion.normalizers import (
    clean_common_noise_lines,
    merge_broken_lines,
    normalize_whitespace,
    remove_repeated_headers_footers,
)
from app.ingestion.parsers.base import is_heading_like, make_block
from app.ingestion.schemas import DocumentBlock, ParsedDocument


def extract_pdf_pages(file_path: str) -> List[Dict]:
    """
    提取 PDF 每一页的原始文本与行信息。

    返回示例：
    [
        {
            "page_num": 1,
            "text": "...",
            "lines": ["...", "..."]
        },
        ...
    ]
    """
    pages: List[Dict] = []

    with pdfplumber.open(file_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            lines = text.split("\n") if text else []

            pages.append(
                {
                    "page_num": idx,
                    "text": text,
                    "lines": lines,
                }
            )

    return pages


def detect_pdf_heading(line: str) -> Tuple[bool, Optional[int]]:
    """
    PDF 标题识别。
    第一版主要靠启发式：
    - 常见编号模式
    - 行较短
    - 非句末标点结束
    """
    normalized = normalize_whitespace(line)
    if not normalized:
        return False, None

    return is_heading_like(normalized)


def lines_to_blocks(page_lines: List[str], page_num: int, start_order: int) -> List[DocumentBlock]:
    """
    将单页文本行转换为结构化 blocks。
    """
    blocks: List[DocumentBlock] = []
    order = start_order

    merged_lines = merge_broken_lines(page_lines)

    for line in merged_lines:
        text = normalize_whitespace(line)
        if not text:
            continue

        is_heading, level = detect_pdf_heading(text)
        block_type = "heading" if is_heading else "paragraph"

        blocks.append(
            make_block(
                block_type=block_type,
                text=text,
                order=order,
                page_num=page_num,
                level=level if is_heading else None,
                prefix="pdf",
                metadata={"page_num": page_num},
            )
        )
        order += 1

    return blocks


def clean_pdf_pages(pages: List[Dict]) -> List[Dict]:
    """
    清洗 PDF 页内容：
    - 去空白
    - 去页码
    - 去纯符号行
    - 去重复页眉页脚
    """
    if not pages:
        return []

    all_page_lines: List[List[str]] = []
    for page in pages:
        lines = page.get("lines", [])
        cleaned_lines = clean_common_noise_lines(lines)
        all_page_lines.append(cleaned_lines)

    cleaned_all_pages = remove_repeated_headers_footers(all_page_lines)

    cleaned_pages: List[Dict] = []
    for page, cleaned_lines in zip(pages, cleaned_all_pages):
        cleaned_pages.append(
            {
                "page_num": page["page_num"],
                "text": "\n".join(cleaned_lines),
                "lines": cleaned_lines,
            }
        )

    return cleaned_pages


def extract_pdf_raw_text(pages: List[Dict]) -> str:
    """
    将 PDF 原始页文本拼接成 raw_text。
    """
    raw_parts: List[str] = []

    for page in pages:
        text = normalize_whitespace(page.get("text", ""))
        if text:
            raw_parts.append(text)

    return "\n\n".join(raw_parts)


def _promote_first_heading_to_title(blocks: List[DocumentBlock]) -> List[DocumentBlock]:
    """
    将第一条 heading 提升为 title，便于后续文档标题提取。
    """
    for block in blocks:
        if block.block_type == "heading" and normalize_whitespace(block.text):
            block.block_type = "title"
            if block.level is None:
                block.level = 1
            return blocks
    return blocks


def parse_pdf_document(file_path: str, source_type: str = "folder") -> ParsedDocument:
    """
    解析 PDF 文档，输出统一的 ParsedDocument。

    第一版能力：
    - 提取 PDF 文本
    - 按页保留 page_num
    - 清理基础噪音（页码 / 页眉页脚 / 空白）
    - 做轻量标题识别
    - 保留段落边界的初步结构
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not path.is_file():
        raise ValueError(f"不是有效文件: {file_path}")

    raw_pages = extract_pdf_pages(str(path))
    cleaned_pages = clean_pdf_pages(raw_pages)

    blocks: List[DocumentBlock] = []
    order = 1

    for page in cleaned_pages:
        page_num = page["page_num"]
        page_lines = page.get("lines", [])

        if not page_lines:
            continue

        page_blocks = lines_to_blocks(page_lines, page_num=page_num, start_order=order)
        blocks.extend(page_blocks)
        order += len(page_blocks)

    blocks = _promote_first_heading_to_title(blocks)

    raw_text = extract_pdf_raw_text(raw_pages)
    clean_text = "\n\n".join(block.text for block in blocks if block.text)

    return ParsedDocument(
        title=path.stem,
        source_type=source_type,
        file_type="pdf",
        source_path=str(path.resolve()),
        raw_text=raw_text,
        clean_text=clean_text,
        blocks=blocks,
        metadata={
            "parser": "pdfplumber",
            "page_count": len(raw_pages),
            "block_count": len(blocks),
        },
    )

