"""
Document ingestion pipeline.

Public API
----------
parse_document(file_path, config=None)
    → ParsedDocument

parse_documents_from_folder(folder_path, config=None)
    → list[ParsedDocument]
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.ingestion.config import ParsingConfig
from app.ingestion.detectors import detect_file_type
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.pdf_parser import PdfParser
from app.ingestion.parsers.text_parser import TextParser
from app.ingestion.schemas import ParsedDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

_PARSER_MAP = {
    "pdf": PdfParser,
    "docx": DocxParser,
    "text": TextParser,
}


def parse_document(
    file_path: str | Path,
    config: ParsingConfig | None = None,
) -> ParsedDocument:
    """
    Parse a single document into a unified ParsedDocument.

    Parameters
    ----------
    file_path : str | Path
        Path to the document file.
    config : ParsingConfig | None
        Parser configuration.  If None, defaults are used.

    Returns
    -------
    ParsedDocument
        Structured document with blocks and metadata.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file type is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    file_type = detect_file_type(path)
    parser_cls = _PARSER_MAP.get(file_type)
    if not parser_cls:
        raise ValueError(f"unsupported file type: {path.suffix!r} (detected as {file_type})")

    parser = parser_cls(config=config)
    return parser.parse(path)


def parse_documents_from_folder(
    folder_path: str | Path,
    config: ParsingConfig | None = None,
    recursive: bool = True,
) -> list[ParsedDocument]:
    """
    Recursively parse all supported documents in a folder.

    Parameters
    ----------
    folder_path : str | Path
    config      : ParsingConfig | None
    recursive   : bool
        If True (default), descend into subdirectories.

    Returns
    -------
    list[ParsedDocument]
        Successfully parsed documents (failures are logged and skipped,
        not raised).
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"folder not found: {folder}")

    results: list[ParsedDocument] = []
    pattern = "**/*" if recursive else "*"

    for path in sorted(folder.glob(pattern)):
        if not path.is_file():
            continue
        try:
            doc = parse_document(path, config=config)
            results.append(doc)
        except Exception as e:
            logger.warning("[pipeline] parse failed: %s | %s", path, e)

    return results
