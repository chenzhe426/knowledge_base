from pathlib import Path
import mimetypes


SUPPORTED_FILE_TYPES = {"txt", "md", "pdf", "docx"}

EXTENSION_TO_FILE_TYPE = {
    ".txt": "txt",
    ".md": "md",
    ".pdf": "pdf",
    ".docx": "docx",
}

MIME_TYPE_TO_FILE_TYPE = {
    "text/plain": "txt",
    "text/markdown": "md",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


def detect_file_type(file_path: str) -> str:
    """
    根据文件路径识别文档类型。

    优先级：
    1. 文件扩展名
    2. mimetypes 推断

    支持：
    - txt
    - md
    - pdf
    - docx
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not path.is_file():
        raise ValueError(f"不是有效文件: {file_path}")

    # 1) 优先按扩展名判断
    suffix = path.suffix.lower()
    if suffix in EXTENSION_TO_FILE_TYPE:
        return EXTENSION_TO_FILE_TYPE[suffix]

    # 2) 再按 mime type 兜底
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type in MIME_TYPE_TO_FILE_TYPE:
        return MIME_TYPE_TO_FILE_TYPE[mime_type]

    raise ValueError(f"暂不支持的文件类型: {suffix or 'unknown'}")


def is_supported_file(file_path: str) -> bool:
    """
    判断文件是否为当前支持的文档类型。
    """
    try:
        file_type = detect_file_type(file_path)
        return file_type in SUPPORTED_FILE_TYPES
    except (FileNotFoundError, ValueError):
        return False


def get_supported_extensions() -> list[str]:
    """
    返回当前支持的文件扩展名列表。
    """
    return sorted(EXTENSION_TO_FILE_TYPE.keys())


def get_supported_file_types() -> list[str]:
    """
    返回当前支持的标准文件类型列表。
    """
    return sorted(SUPPORTED_FILE_TYPES)


def guess_source_type(source: str) -> str:
    """
    粗略判断输入来源类型。

    返回：
    - url
    - folder
    - upload

    说明：
    当前第一版主要用于给 pipeline 打标签，后续可扩展得更细。
    """
    if not source:
        return "upload"

    source_lower = source.lower().strip()

    if source_lower.startswith("http://") or source_lower.startswith("https://"):
        return "url"

    path = Path(source)
    if path.exists() and path.is_dir():
        return "folder"

    return "upload"