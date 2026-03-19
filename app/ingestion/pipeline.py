from __future__ import annotations

from pathlib import Path

from app.ingestion.detectors import detect_file_type
from app.ingestion.parsers import DocxParser, ParsedDocument, PdfParser, TextParser


def parse_document(file_path: str | Path) -> ParsedDocument:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    file_type = detect_file_type(path)

    if file_type == "text":
        parser = TextParser()
    elif file_type == "docx":
        parser = DocxParser()
    elif file_type == "pdf":
        parser = PdfParser()
    else:
        raise ValueError(f"unsupported file type: {path.suffix}")

    return parser.parse(path)


def parse_documents_from_folder(folder_path: str | Path) -> list[ParsedDocument]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"folder not found: {folder}")

    results: list[ParsedDocument] = []

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        try:
            doc = parse_document(path)
            results.append(doc)
        except Exception as e:
            print(f"[WARN] parse failed: {path} | {e}")

    return results