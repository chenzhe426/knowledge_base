import re
from collections import Counter
from typing import Iterable, List

from app.ingestion.schemas import DocumentBlock


PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*第?\s*\d+\s*页\s*$"),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*\d+\s*$"),
]

SYMBOL_ONLY_PATTERN = re.compile(r"^[\W_]+$", re.UNICODE)


def normalize_whitespace(text: str) -> str:
    """
    统一空白字符格式。
    - 全角空格转半角
    - Windows 换行转 Unix 换行
    - 压缩连续空格 / 制表符
    - 压缩 3 个以上连续空行为 2 个
    """
    if not text:
        return ""

    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_page_number_line(line: str) -> bool:
    """
    判断一行是否像页码。
    """
    if not line:
        return False

    stripped = line.strip()
    if not stripped:
        return False

    return any(pattern.match(stripped) for pattern in PAGE_NUMBER_PATTERNS)


def is_symbol_only_line(line: str) -> bool:
    """
    判断一行是否基本只有符号。
    """
    if not line:
        return False

    stripped = line.strip()
    if not stripped:
        return False

    return bool(SYMBOL_ONLY_PATTERN.match(stripped))


def clean_common_noise_lines(lines: List[str]) -> List[str]:
    """
    清理通用噪音行：
    - 空行
    - 页码
    - 纯符号行
    """
    cleaned = []
    for line in lines:
        normalized = normalize_whitespace(line)
        if not normalized:
            continue
        if is_page_number_line(normalized):
            continue
        if is_symbol_only_line(normalized):
            continue
        cleaned.append(normalized)
    return cleaned


def _line_frequency(lines: Iterable[str]) -> Counter:
    """
    统计行频次，忽略空行。
    """
    counter = Counter()
    for line in lines:
        normalized = normalize_whitespace(line)
        if normalized:
            counter[normalized] += 1
    return counter


def remove_repeated_headers_footers(pages: List[List[str]], min_repeat_pages: int = 2) -> List[List[str]]:
    """
    移除 PDF 中高频重复的页眉 / 页脚候选。

    规则：
    - 只看每页前 2 行和后 2 行
    - 在多页重复出现的短文本，视为页眉/页脚候选
    - 页码模式也会被移除

    参数：
    - pages: [[page1_line1, page1_line2, ...], [page2_line1, ...], ...]
    - min_repeat_pages: 至少重复出现多少页才算噪音
    """
    if not pages:
        return []

    edge_lines = []
    for page_lines in pages:
        cleaned_page = [normalize_whitespace(line) for line in page_lines if normalize_whitespace(line)]
        if not cleaned_page:
            continue
        edge_lines.extend(cleaned_page[:2])
        edge_lines.extend(cleaned_page[-2:])

    freq = _line_frequency(edge_lines)
    repeated_noise = {
        line
        for line, count in freq.items()
        if count >= min_repeat_pages and len(line) <= 80
    }

    cleaned_pages = []
    for page_lines in pages:
        normalized_lines = [normalize_whitespace(line) for line in page_lines]
        normalized_lines = [line for line in normalized_lines if line]

        if not normalized_lines:
            cleaned_pages.append([])
            continue

        page_cleaned = []
        for idx, line in enumerate(normalized_lines):
            is_edge = idx < 2 or idx >= len(normalized_lines) - 2
            if is_page_number_line(line):
                continue
            if is_edge and line in repeated_noise:
                continue
            page_cleaned.append(line)

        cleaned_pages.append(page_cleaned)

    return cleaned_pages


def _looks_like_heading(line: str) -> bool:
    """
    一个轻量级启发式判断，用于段落拼接时避免把标题并进正文。
    """
    line = normalize_whitespace(line)
    if not line:
        return False

    if len(line) <= 30 and re.match(r"^#{1,6}\s+", line):
        return True

    heading_patterns = [
        r"^第[一二三四五六七八九十\d]+[章节部分篇]",
        r"^[一二三四五六七八九十]+、",
        r"^（[一二三四五六七八九十\d]+）",
        r"^\(\d+\)",
        r"^\d+(\.\d+){0,3}\s+",
    ]
    if any(re.match(pattern, line) for pattern in heading_patterns):
        return True

    if len(line) <= 25 and not re.search(r"[。！？；：:]$", line):
        return True

    return False


def _should_attach_to_previous(prev: str, curr: str) -> bool:
    """
    判断 curr 是否应与 prev 合并为同一段。
    """
    prev = normalize_whitespace(prev)
    curr = normalize_whitespace(curr)

    if not prev or not curr:
        return False

    if _looks_like_heading(curr):
        return False

    if re.match(r"^[-•*]\s+", curr):
        return False

    if re.match(r"^\d+[.)、]\s+", curr):
        return False

    # 上一行像未结束句子，则更可能接到下一行
    if not re.search(r"[。！？；：:.!?]$", prev):
        return True

    # 上一行很短且当前行不似标题，也倾向合并
    if len(prev) < 20 and not _looks_like_heading(curr):
        return True

    return False


def merge_broken_lines(lines: List[str]) -> List[str]:
    """
    将 PDF 等文档中被错误断开的文本行重新合并成更自然的段落。
    """
    normalized_lines = [normalize_whitespace(line) for line in lines]
    normalized_lines = [line for line in normalized_lines if line]

    if not normalized_lines:
        return []

    merged = [normalized_lines[0]]

    for line in normalized_lines[1:]:
        prev = merged[-1]
        if _should_attach_to_previous(prev, line):
            merged[-1] = f"{prev} {line}".strip()
        else:
            merged.append(line)

    return merged


def build_clean_text_from_blocks(blocks: List[DocumentBlock]) -> str:
    """
    根据结构化 blocks 重建 clean_text。
    标题与正文之间保留段落边界，便于后续 chunking。
    """
    if not blocks:
        return ""

    pieces: List[str] = []

    for block in sorted(blocks, key=lambda b: b.order):
        text = normalize_whitespace(block.text)
        if not text:
            continue

        if block.block_type in {"title", "heading"}:
            if pieces and pieces[-1] != "":
                pieces.append("")
            pieces.append(text)
            pieces.append("")
        else:
            pieces.append(text)
            pieces.append("")

    clean_text = "\n".join(pieces)
    clean_text = normalize_whitespace(clean_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    return clean_text.strip()


def extract_title_from_blocks(blocks: List[DocumentBlock], fallback: str) -> str:
    """
    优先从 blocks 中提取标题：
    1. 第一个 title
    2. 第一个 heading
    3. fallback
    """
    for block in sorted(blocks, key=lambda b: b.order):
        if block.block_type == "title" and normalize_whitespace(block.text):
            return normalize_whitespace(block.text)

    for block in sorted(blocks, key=lambda b: b.order):
        if block.block_type == "heading" and normalize_whitespace(block.text):
            return normalize_whitespace(block.text)

    return fallback.strip() if fallback else "Untitled"

