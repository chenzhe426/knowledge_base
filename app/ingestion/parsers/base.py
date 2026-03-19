import re
from itertools import count
from pathlib import Path
from typing import Optional, Tuple

from app.ingestion.normalizers import normalize_whitespace
from app.ingestion.schemas import DocumentBlock


_block_counter = count(1)


def next_block_id(prefix: str = "block", order: Optional[int] = None) -> str:
    """
    生成稳定可读的 block_id。

    示例：
    - txt_1
    - pdf_12
    - block_7
    """
    idx = order if order is not None else next(_block_counter)
    return f"{prefix}_{idx}"


def make_block(
    *,
    block_type: str,
    text: str,
    order: int,
    page_num: Optional[int] = None,
    level: Optional[int] = None,
    prefix: str = "block",
    metadata: Optional[dict] = None,
) -> DocumentBlock:
    """
    统一创建 DocumentBlock，避免各 parser 重复拼字段。
    """
    normalized_text = normalize_whitespace(text)

    return DocumentBlock(
        block_id=next_block_id(prefix=prefix, order=order),
        block_type=block_type,
        text=normalized_text,
        order=order,
        page_num=page_num,
        level=level,
        metadata=metadata or {},
    )


def safe_read_bytes(file_path: str) -> bytes:
    """
    安全读取文件二进制内容。
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not path.is_file():
        raise ValueError(f"不是有效文件: {file_path}")

    return path.read_bytes()


def infer_heading_level(text: str) -> Optional[int]:
    """
    根据常见标题模式粗略推断标题层级。
    这是启发式规则，不保证 100% 准确。
    """
    line = normalize_whitespace(text)
    if not line:
        return None

    patterns = [
        (1, r"^第[一二三四五六七八九十百千\d]+[章节部分篇]\b"),
        (1, r"^[一二三四五六七八九十百千]+、"),
        (2, r"^（[一二三四五六七八九十百千\d]+）"),
        (2, r"^\([一二三四五六七八九十百千\d]+\)"),
        (2, r"^\d+\s+[^\d]"),
        (2, r"^\d+[.)、]"),
        (2, r"^\d+\.\d+\s+"),
        (3, r"^\d+\.\d+\.\d+\s+"),
        (4, r"^\d+\.\d+\.\d+\.\d+\s+"),
        (1, r"^#{1}\s+"),
        (2, r"^#{2}\s+"),
        (3, r"^#{3}\s+"),
        (4, r"^#{4,6}\s+"),
    ]

    for level, pattern in patterns:
        if re.match(pattern, line):
            return level

    return None


def is_heading_like(text: str) -> Tuple[bool, Optional[int]]:
    """
    通用启发式标题识别。

    返回：
    - bool: 是否像标题
    - Optional[int]: 推断的层级
    """
    line = normalize_whitespace(text)
    if not line:
        return False, None

    level = infer_heading_level(line)
    if level is not None:
        return True, level

    # 很短、独立、且不像完整句子，常常是标题
    if len(line) <= 25 and not re.search(r"[。！？；：:.!?]$", line):
        return True, 2

    # 中等长度且没有明显句末标点，也可能是标题
    if len(line) <= 40 and not re.search(r"[。！？；：:.!?]$", line):
        return True, 3

    return False, None

