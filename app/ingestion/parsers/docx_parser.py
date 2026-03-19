from pathlib import Path
from typing import List, Optional, Tuple

from docx import Document

from app.ingestion.normalizers import normalize_whitespace
from app.ingestion.parsers.base import is_heading_like, make_block
from app.ingestion.schemas import DocumentBlock, ParsedDocument


def detect_docx_block_type(style_name: str, text: str) -> Tuple[str, Optional[int]]:
    """
    根据 Word 段落样式判断 block 类型与标题层级。

    规则优先级：
    1. Title / Subtitle
    2. Heading 1~6
    3. 再用启发式规则兜底
    """
    normalized_style = (style_name or "").strip().lower()
    normalized_text = normalize_whitespace(text)

    if not normalized_text:
        return "paragraph", None

    if "title" == normalized_style or normalized_style.endswith(" title"):
        return "title", 1

    if "subtitle" in normalized_style:
        return "heading", 2

    if "heading" in normalized_style:
        # 常见样式名：Heading 1 / heading 2 / 标题 1（后续可继续扩展）
        for level in range(1, 7):
            if f"heading {level}" in normalized_style or f"标题 {level}" in normalized_style:
                return ("title", 1) if level == 1 else ("heading", level)

        # 样式名里有 heading 但没识别出具体层级
        return "heading", 2

    # 样式无法识别时，走轻量启发式
    heading_like, level = is_heading_like(normalized_text)
    if heading_like:
        if level == 1:
            return "title", 1
        return "heading", level or 2

    return "paragraph", None


def extract_docx_raw_text(doc: Document) -> str:
    """
    提取 docx 中所有非空段落，拼成原始文本。
    """
    lines: List[str] = []
    for para in doc.paragraphs:
        text = normalize_whitespace(para.text)
        if text:
            lines.append(text)
    return "\n\n".join(lines)


def parse_docx_paragraphs(doc: Document) -> List[DocumentBlock]:
    """
    遍历 docx 段落并生成结构化 blocks。
    """
    blocks: List[DocumentBlock] = []
    order = 1
    title_already_used = False

    for para in doc.paragraphs:
        text = normalize_whitespace(para.text)
        if not text:
            continue

        style_name = para.style.name if para.style is not None else ""
        block_type, level = detect_docx_block_type(style_name, text)

        # 避免多个 block 都被打成 title
        if block_type == "title":
            if title_already_used:
                block_type = "heading"
                level = level or 1
            else:
                title_already_used = True

        blocks.append(
            make_block(
                block_type=block_type,
                text=text,
                order=order,
                level=level,
                prefix="docx",
                metadata={"style_name": style_name},
            )
        )
        order += 1

    return blocks


def parse_docx_document(file_path: str, source_type: str = "folder") -> ParsedDocument:
    """
    解析 docx 文档，输出统一的 ParsedDocument。

    第一版目标：
    - 提取段落文本
    - 识别 Word 原生标题样式
    - 保留段落边界
    - 输出统一 blocks
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not path.is_file():
        raise ValueError(f"不是有效文件: {file_path}")

    doc = Document(str(path))

    raw_text = extract_docx_raw_text(doc)
    blocks = parse_docx_paragraphs(doc)
    clean_text = "\n\n".join(block.text for block in blocks if block.text)

    return ParsedDocument(
        title=path.stem,
        source_type=source_type,
        file_type="docx",
        source_path=str(path.resolve()),
        raw_text=raw_text,
        clean_text=clean_text,
        blocks=blocks,
        metadata={
            "parser": "python-docx",
            "paragraph_count": len(blocks),
        },
    )
