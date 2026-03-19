from pathlib import Path
from typing import List

from app.ingestion.detectors import detect_file_type
from app.ingestion.loaders import load_file_paths_from_folder, load_single_file
from app.ingestion.normalizers import build_clean_text_from_blocks, extract_title_from_blocks
from app.ingestion.parsers.docx_parser import parse_docx_document
from app.ingestion.parsers.pdf_parser import parse_pdf_document
from app.ingestion.parsers.text_parser import parse_text_document
from app.ingestion.schemas import ParsedDocument


def parse_document(file_path: str, source_type: str = "folder") -> ParsedDocument:
    """
    统一文档解析入口。

    流程：
    1. 校验文件路径
    2. 识别文件类型
    3. 路由到对应 parser
    4. 基于 blocks 重建 clean_text
    5. 自动补标题

    支持：
    - txt
    - md
    - pdf
    - docx
    """
    resolved_path = load_single_file(file_path)
    file_type = detect_file_type(resolved_path)

    if file_type in {"txt", "md"}:
        parsed = parse_text_document(resolved_path, source_type=source_type)
    elif file_type == "docx":
        parsed = parse_docx_document(resolved_path, source_type=source_type)
    elif file_type == "pdf":
        parsed = parse_pdf_document(resolved_path, source_type=source_type)
    else:
        raise ValueError(f"暂不支持的文件类型: {file_type}")

    parsed.clean_text = build_clean_text_from_blocks(parsed.blocks)
    parsed.title = extract_title_from_blocks(parsed.blocks, fallback=Path(resolved_path).stem)
    parsed.source_path = resolved_path
    parsed.source_type = source_type
    parsed.file_type = file_type

    if not parsed.raw_text:
        parsed.raw_text = parsed.clean_text

    return parsed


def parse_documents_from_folder(folder_path: str, recursive: bool = True) -> List[ParsedDocument]:
    """
    解析文件夹中的所有支持文件。

    参数：
    - folder_path: 文件夹路径
    - recursive: 是否递归扫描子目录

    返回：
    - ParsedDocument 列表
    """
    file_paths = load_file_paths_from_folder(folder_path, recursive=recursive)
    parsed_documents: List[ParsedDocument] = []

    for file_path in file_paths:
        parsed = parse_document(file_path, source_type="folder")
        parsed_documents.append(parsed)

    return parsed_documents

