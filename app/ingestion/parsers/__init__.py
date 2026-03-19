from .text_parser import parse_text_document
from .docx_parser import parse_docx_document
from .pdf_parser import parse_pdf_document

__all__ = [
    "parse_text_document",
    "parse_docx_document",
    "parse_pdf_document",
]


