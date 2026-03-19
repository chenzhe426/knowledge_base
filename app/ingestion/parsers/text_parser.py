from pathlib import Path
from typing import List, Tuple

from app.ingestion.normalizers import normalize_whitespace
from app.ingestion.parsers.base import make_block
from app.ingestion.schemas import DocumentBlock, ParsedDocument


def read_text_with_fallback(file_path: str) -> str:
    """
    以常见编码顺序读取文本文件。
    优先尝试 utf-8 / utf-8-sig / gbk / gb18030 / latin-1。
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not path.is_file():
        raise ValueError(f"不是有效文件: {file_path}")

    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"]
    last_error = None

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"无法解码文本文件: {file_path}, 最后一次错误: {last_error}",
    )


def parse_markdown_heading(line: str) -> Tuple[bool, int | None, str]:
    """
    识别 Markdown 标题。
    返回：
    - 是否是标题
    - 标题层级
    - 去掉 # 之后的正文
    """
    stripped = line.strip()
    if not stripped.startswith("#"):
        return False, None, stripped

    level = 0
    for ch in stripped:
        if ch == "#":
            level += 1
        else:
            break

    if 1 <= level <= 6 and len(stripped) > level and stripped[level] == " ":
        return True, level, stripped[level + 1 :].strip()

    return False, None, stripped


def split_text_into_paragraphs(text: str) -> List[str]:
    """
    按空行切分纯文本段落，保留段落边界。
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = normalized.split("\n\n")
    paragraphs: List[str] = []

    for part in parts:
        paragraph = normalize_whitespace(part)
        if paragraph:
            paragraphs.append(paragraph)

    return paragraphs


def parse_plain_text_paragraphs(text: str) -> List[DocumentBlock]:
    """
    将普通 txt 文本按段落切成 paragraph blocks。
    """
    paragraphs = split_text_into_paragraphs(text)
    blocks: List[DocumentBlock] = []

    for idx, paragraph in enumerate(paragraphs, start=1):
        blocks.append(
            make_block(
                block_type="paragraph",
                text=paragraph,
                order=idx,
                prefix="txt",
            )
        )

    return blocks


def parse_markdown_lines(lines: List[str]) -> List[DocumentBlock]:
    """
    解析 Markdown：
    - 识别 # / ## / ### 标题
    - 其他内容按段落聚合
    """
    blocks: List[DocumentBlock] = []
    paragraph_buffer: List[str] = []
    order = 1

    def flush_paragraph() -> None:
        nonlocal order
        if not paragraph_buffer:
            return

        paragraph_text = normalize_whitespace(" ".join(paragraph_buffer))
        if paragraph_text:
            blocks.append(
                make_block(
                    block_type="paragraph",
                    text=paragraph_text,
                    order=order,
                    prefix="md",
                )
            )
            order += 1
        paragraph_buffer.clear()

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            continue

        is_heading, level, heading_text = parse_markdown_heading(stripped)
        if is_heading:
            flush_paragraph()
            block_type = "title" if level == 1 and not blocks else "heading"
            blocks.append(
                make_block(
                    block_type=block_type,
                    text=heading_text,
                    order=order,
                    level=level,
                    prefix="md",
                )
            )
            order += 1
            continue

        paragraph_buffer.append(stripped)

    flush_paragraph()
    return blocks


def parse_text_document(file_path: str, source_type: str = "folder") -> ParsedDocument:
    """
    解析 txt / md 文档，输出统一的 ParsedDocument。
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    raw_text = read_text_with_fallback(file_path)
    normalized_raw_text = normalize_whitespace(raw_text)

    if suffix == ".md":
        lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        blocks = parse_markdown_lines(lines)
        file_type = "md"
    else:
        blocks = parse_plain_text_paragraphs(raw_text)
        file_type = "txt"

    clean_text = "\n\n".join(block.text for block in blocks if block.text)

    return ParsedDocument(
        title=path.stem,
        source_type=source_type,
        file_type=file_type,
        source_path=str(path.resolve()),
        raw_text=normalized_raw_text,
        clean_text=clean_text,
        blocks=blocks,
        metadata={},
    )

