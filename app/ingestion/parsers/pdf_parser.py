"""
PDF parser with multi-strategy selection and unified block output.

Strategies (in order of preference)
------------------------------------
1. **docling**        — best structure extraction (headings, tables, layout)
2. **pdfplumber**     — reliable plain-text fallback with coordinate access
3. **pypdfium2**      — PyMuPDF-based, good for text ordering in complex layouts

For each candidate that succeeds we:
  a) Run quality scoring on the resulting blocks
  b) Pick the highest-scoring candidate that meets the minimum threshold
  c) Record why that candidate was chosen (selection_reason)

PDF-specific processing
-----------------------
- Heading detection via font-size heuristics (pdfplumber path)
- Table detection and preservation as ``table`` blocks
- Page header/footer removal via coordinate filtering
- Hyphenation repair
- ``likely_scanned_pdf`` + ``ocr_recommended`` flags when quality is poor
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from app.ingestion.config import CleaningConfig, PdfParserConfig, ParsingConfig
from app.ingestion.loaders import load_binary_file
from app.ingestion.normalizers import (
    blocks_to_content,
    clean_blocks,
    normalize_block_text,
    normalize_text,
)
from app.ingestion.parsers.base import BaseParser
from app.ingestion.quality import score_pdf_blocks
from app.ingestion.schemas import ParsedDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section path helper (shared within this module)
# ---------------------------------------------------------------------------


def _section_path_push(section_stack: list[str], heading_text: str, level: int) -> str:
    while len(section_stack) >= level:
        section_stack.pop()
    section_stack.append(heading_text)
    return " > ".join(section_stack)


# ---------------------------------------------------------------------------
# Strategy 1 – docling
# ---------------------------------------------------------------------------


def _parse_with_docling(path: Path) -> list[dict[str, Any]]:
    """
    Use docling for high-quality structured PDF extraction.

    docling exports to markdown, which we then parse into blocks while
    preserving headings, tables, and list items.
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    markdown_text = result.document.export_to_markdown()

    lines = normalize_text(markdown_text).split("\n")
    blocks: list[dict[str, Any]] = []
    section_stack: list[str] = []
    para_buf: list[str] = []

    def flush_paragraph():
        nonlocal para_buf
        if para_buf:
            text = "\n".join(para_buf).strip()
            if text:
                blocks.append(
                    {
                        "type": "paragraph",
                        "text": text,
                        "section_path": " > ".join(section_stack) if section_stack else None,
                        "parser_source": "docling",
                    }
                )
            para_buf = []

    for line in lines:
        s = line.strip()
        if not s:
            flush_paragraph()
            continue

        # Markdown heading
        if s.startswith("#"):
            flush_paragraph()
            level = len(s) - len(s.lstrip("#"))
            heading_text = s[level:].strip()
            section_path = _section_path_push(section_stack, heading_text, level)
            blocks.append(
                {
                    "type": "heading",
                    "text": heading_text,
                    "heading_level": level,
                    "section_path": section_path,
                    "parser_source": "docling",
                }
            )
            continue

        # Table row
        if s.startswith("|") and s.endswith("|"):
            flush_paragraph()
            blocks.append(
                {
                    "type": "table",
                    "text": s,
                    "section_path": " > ".join(section_stack) if section_stack else None,
                    "parser_source": "docling",
                }
            )
            continue

        # List item
        if re.match(r"^\s*([-*•·]|\d+[.)、])\s+.+$", s):
            flush_paragraph()
            blocks.append(
                {
                    "type": "list_item",
                    "text": s,
                    "section_path": " > ".join(section_stack) if section_stack else None,
                    "parser_source": "docling",
                }
            )
            continue

        para_buf.append(s)

    flush_paragraph()
    return blocks


# ---------------------------------------------------------------------------
# Strategy 2 – pdfplumber
# ---------------------------------------------------------------------------


def _parse_with_pdfplumber(path: Path, config: PdfParserConfig) -> list[dict[str, Any]]:
    """
    Parse PDF using pdfplumber with per-page text extraction.

    We extract:
    - Text blocks with bounding-box coordinates
    - Tables (via page.extract_tables())
    - Per-word font-size information for heading detection

    Heading detection heuristic:
      Words with font-size > heading_font_size_threshold (default 12pt)
      and all-caps ratio > 0.5 are treated as potential headings.
    """
    import pdfplumber

    blocks: list[dict[str, Any]] = []
    section_stack: list[str] = []

    with pdfplumber.open(str(path)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            page_height = page.height

            # Extract tables
            tables = page.extract_tables()
            table_texts: set[int] = set()

            if tables and config.extract_tables:
                for table_rows in tables:
                    if not table_rows:
                        continue
                    table_lines: list[str] = []
                    for row in table_rows:
                        if not row:
                            continue
                        # Filter out None cells
                        cells = [str(c).strip() if c is not None else "" for c in row]
                        table_lines.append(" | ".join(cells))
                    if table_lines:
                        table_md = "\n".join(table_lines)
                        blocks.append(
                            {
                                "type": "table",
                                "text": table_md,
                                "page": page_idx,
                                "section_path": None,
                                "parser_source": "pdfplumber",
                                "bbox": None,
                            }
                        )

            # Extract words with positions
            words = page.extract_words()
            if not words:
                continue

            # Build text from words, preserving approximate layout
            # Group by vertical position (same "line")
            lines_by_y: dict[float, list[dict]] = {}
            for w in words:
                top = round(w["top"], 1)
                if top not in lines_by_y:
                    lines_by_y[top] = []
                lines_by_y[top].append(w)

            for y_pos, line_words in sorted(lines_by_y.items()):
                # Sort by x position
                line_words.sort(key=lambda w: w["x0"])
                line_text = " ".join(w["text"] for w in line_words).strip()
                if not line_text:
                    continue

                # Font-size heuristics for heading detection
                font_sizes = [w.get("font_size", 0) for w in line_words if w.get("font_size")]
                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                allcaps_ratio = sum(1 for c in line_text if c.isupper()) / max(len(line_text), 1)

                # Determine block type
                btype = "paragraph"
                heading_level = None
                section_path = None

                # Coordinate-based header/footer removal
                if config.remove_header_footer_by_coords:
                    dist_from_top = y_pos
                    dist_from_bottom = page_height - y_pos
                    if (
                        dist_from_top < config.header_footer_y_threshold
                        or dist_from_bottom < config.header_footer_y_threshold
                    ):
                        btype = "header" if dist_from_top < config.header_footer_y_threshold else "footer"

                # Font-size heading detection
                if (
                    avg_font_size >= config.heading_font_size_threshold
                    and allcaps_ratio > 0.5
                    and len(line_text) < 120
                ):
                    btype = "heading"
                    heading_level = 1
                    # Try to detect level from font size
                    if avg_font_size >= 18:
                        heading_level = 1
                    elif avg_font_size >= 14:
                        heading_level = 2
                    else:
                        heading_level = 3
                    section_path = _section_path_push(section_stack, line_text, heading_level)
                elif (
                    avg_font_size >= config.heading_font_size_threshold * 1.2
                    and len(line_text) < 80
                    and btype == "paragraph"
                ):
                    # Possible subheading
                    btype = "heading"
                    heading_level = 4
                    section_path = _section_path_push(section_stack, line_text, heading_level)

                # Bounding box (x0, y0, x1, y1)
                x0 = min(w["x0"] for w in line_words)
                x1 = max(w["x1"] for w in line_words)
                bbox = (round(x0, 1), round(y_pos, 1), round(x1, 1), round(w["bottom"], 1))

                if btype == "paragraph":
                    section_path = " > ".join(section_stack) if section_stack else None

                blocks.append(
                    {
                        "type": btype,
                        "text": line_text,
                        "page": page_idx,
                        "heading_level": heading_level,
                        "section_path": section_path,
                        "parser_source": "pdfplumber",
                        "bbox": bbox,
                    }
                )

    return blocks


# ---------------------------------------------------------------------------
# Strategy 3 – pypdfium2 (PyMuPDF-compatible)
# ---------------------------------------------------------------------------


def _parse_with_pymupdf(path: Path, config: PdfParserConfig) -> list[dict[str, Any]]:
    """
    Parse PDF using pypdfium2 for raw text extraction with layout hints.

    pypdfium2 gives us page-level text with fairly reliable reading order.
    """
    import pypdfium2 as pdfium

    blocks: list[dict[str, Any]] = []
    section_stack: list[str] = []

    pdf = pdfium.PdfDocument(str(path))
    page_count = len(pdf)

    for page_idx in range(page_count):
        page = pdf[page_idx]
        page_height = page.get_height_point()
        page_width = page.get_width_point()

        # Get text page for structured access
        textpage = page.get_textpage()
        blocks_bynode = textpage.get_text_blocks()

        for node in blocks_bynode:
            if not node:
                continue

            # node is a dict-like with keys: type, string, origin (x,y)
            node_type = node.get("type", 0)  # 0=text, 4=placeholder
            if node_type not in (0,):  # skip non-text items
                continue

            text = node.get("string", "").strip()
            if not text:
                continue

            origin = node.get("origin", (0, 0))
            x, y = origin.x if hasattr(origin, "x") else origin[0], origin.y if hasattr(origin, "y") else origin[1]

            # Coordinate-based header/footer removal
            btype = "paragraph"
            heading_level = None
            section_path = None
            bbox = None

            if config.remove_header_footer_by_coords:
                dist_top = y
                dist_bottom = page_height - y
                if dist_top < config.header_footer_y_threshold:
                    btype = "header"
                elif dist_bottom < config.header_footer_y_threshold:
                    btype = "footer"

            # Basic heading detection: all caps + short
            # (pypdfium2 doesn't expose font_size easily without deeper API calls)
            if btype == "paragraph":
                allcaps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
                if allcaps_ratio > 0.8 and len(text) < 100:
                    btype = "heading"
                    heading_level = 1
                    section_path = _section_path_push(section_stack, text, 1)
                else:
                    section_path = " > ".join(section_stack) if section_stack else None

            # Build bbox from node info if available
            try:
                bbox = (round(x, 1), round(y, 1), round(x + 10, 1), round(y + 10, 1))
            except Exception:
                bbox = None

            blocks.append(
                {
                    "type": btype,
                    "text": text,
                    "page": page_idx + 1,
                    "heading_level": heading_level,
                    "section_path": section_path,
                    "parser_source": "pymupdf",
                    "bbox": bbox,
                }
            )

    return blocks


# ---------------------------------------------------------------------------
# Candidate runner & scorer
# ---------------------------------------------------------------------------


def _run_pdf_candidate(
    parser_name: str,
    parser_func,
    path: Path,
    config: PdfParserConfig,
) -> dict[str, Any] | None:
    """Run a single PDF parser candidate and return blocks + score."""
    errors: list[str] = []
    try:
        blocks_or_fn = parser_func(path, config) if parser_name != "docling" else parser_func(path)
        blocks = blocks_or_fn  # both return list[dict]
        if not blocks:
            return None

        score_result = score_pdf_blocks(blocks, config)
        return {
            "parser_name": parser_name,
            "blocks": blocks,
            "quality_score": score_result["quality_score"],
            "score_details": score_result,
            "errors": [],
        }
    except Exception as e:
        errors.append(f"{parser_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Scanned / image-based PDF detection
# ---------------------------------------------------------------------------


def _analyze_pdf_file(path: Path) -> tuple[bool, int, int]:
    """
    Try to determine if a PDF is image-only (scanned) by checking:
    1. File size vs page count ratio (scanned PDFs are typically large)
    2. Whether pypdfium2 can report any text at all

    Returns (likely_scanned, page_count, file_size_bytes).
    """
    file_size = path.stat().st_size if path.exists() else 0
    page_count = 0

    try:
        import pypdfium2
        pdf = pypdfium2.PdfDocument(str(path))
        page_count = len(pdf)

        # Check: large file + relatively few pages → likely scanned
        if page_count > 0:
            avg_bytes_per_page = file_size / page_count
            # Heuristic: > 500 KB/page suggests image-based
            likely_by_size = avg_bytes_per_page > 500 * 1024

            # Also try to extract *any* text from first page
            has_any_text = False
            for page_idx in range(min(3, page_count)):
                page = pdf[page_idx]
                textpage = page.get_textpage()
                text = textpage.read_text()
                if text and text.strip():
                    has_any_text = True
                    break

            likely_scanned = likely_by_size and not has_any_text
            return likely_scanned, page_count, file_size
        else:
            return False, 0, file_size

    except Exception:
        return False, page_count, file_size


# ---------------------------------------------------------------------------
# PDF Parser – public interface
# ---------------------------------------------------------------------------


class PdfParser(BaseParser):
    """
    Multi-strategy PDF parser that scores candidates and selects the best.

    The ``parse()`` method runs all enabled strategies, scores each result,
    picks the highest-quality candidate above the minimum threshold, and
    returns a ``ParsedDocument`` with complete metadata.
    """

    file_type = "pdf"

    def __init__(self, config: ParsingConfig | None = None):
        self._cfg = config.pdf if config else PdfParserConfig()
        self._clean_cfg = config.cleaning if config else CleaningConfig()

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        candidates: dict[str, Any] = {}
        all_errors: list[str] = []

        # ---- Run enabled strategies ----
        strategies: list[tuple[str, Any]] = []
        if self._cfg.use_docling:
            strategies.append(("docling", _parse_with_docling))
        if self._cfg.use_pdfplumber:
            strategies.append(("pdfplumber", lambda p: _parse_with_pdfplumber(p, self._cfg)))
        if self._cfg.use_pymupdf:
            strategies.append(("pymupdf", lambda p: _parse_with_pymupdf(p, self._cfg)))

        for name, fn in strategies:
            result = _run_pdf_candidate(name, fn, path, self._cfg)
            if result:
                candidates[name] = result
            else:
                all_errors.append(f"{name}: no blocks returned")

        if not candidates:
            # All parsers returned 0 blocks — likely a scanned (image-only) PDF.
            # Do file-level analysis to give a helpful error / empty-but-labeled result.
            likely_scanned, page_count, file_size = _analyze_pdf_file(path)

            if likely_scanned:
                # Return an empty ParsedDocument with clear metadata instead of crashing.
                # Downstream can check metadata["likely_scanned_pdf"] to decide what to do.
                logger.warning(
                    "[PdfParser] %s: no text extracted (likely scanned/image PDF). "
                    "file_size=%.1fMB, pages=%d",
                    path, file_size / 1024 / 1024, page_count,
                )
                return ParsedDocument(
                    title=path.stem,
                    content="",
                    blocks=[],
                    source_path=str(path),
                    file_type=self.file_type,
                    metadata={
                        "parser_used": "none",
                        "candidate_scores": {},
                        "selection_reason": "all parsers returned 0 blocks",
                        "quality_score": 0.0,
                        "page_count": page_count,
                        "file_size_bytes": file_size,
                        "table_count": 0,
                        "heading_count": 0,
                        "likely_scanned_pdf": True,
                        "ocr_recommended": True,
                        "source_format": "pdf",
                        "source_file": str(path),
                        "errors": all_errors,
                        "hint": (
                            "This PDF appears to be a scanned image-based document with no "
                            "extractable text. To process it, run OCR first (e.g. pytesseract "
                            "or Adobe Acrobat) and re-export as a text-based PDF, or use a "
                            "PDF with an embedded text layer."
                        ),
                    },
                )

            raise RuntimeError(
                f"all PDF parsers failed for {path} (no text extracted, not a typical "
                f"scanned PDF either — file may be corrupted or password-protected). "
                f"Errors: {all_errors}"
            )

        # ---- Select best candidate ----
        best_name = max(candidates, key=lambda k: candidates[k]["quality_score"])
        best = candidates[best_name]
        blocks = best["blocks"]
        score_details = best["score_details"]

        if self._cfg.verbose:
            logger.info(
                "[PdfParser] strategy scores – %s",
                {k: v["quality_score"] for k, v in candidates.items()},
            )

        # ---- Post-process blocks ----
        for block in blocks:
            block["source_format"] = "pdf"
            # Ensure page is set
            if block.get("page") is None:
                block["page"] = None

        cleaned_blocks = clean_blocks(blocks, self._clean_cfg)

        # Ensure every block has source_format
        for b in cleaned_blocks:
            b.setdefault("source_format", "pdf")

        content = blocks_to_content(cleaned_blocks, include_headings=self._clean_cfg.include_headings_in_content)

        # ---- Aggregate document-level stats ----
        table_count = sum(1 for b in cleaned_blocks if b.get("type") == "table")
        heading_count = sum(1 for b in cleaned_blocks if b.get("type") == "heading")
        page_count = len({b.get("page") for b in cleaned_blocks if b.get("page") is not None})

        # Page count from raw bytes as fallback
        if page_count == 0:
            try:
                import pypdfium2
                pdf = pypdfium2.PdfDocument(str(path))
                page_count = len(pdf)
            except Exception:
                page_count = 0

        metadata = {
            "parser_used": best_name,
            "candidate_scores": {
                k: v["quality_score"] for k, v in candidates.items()
            },
            "selection_reason": (
                f"score={score_details['quality_score']:.3f} vs "
                f"min={self._cfg.min_quality_score}; "
                f"chars={score_details['total_chars']}, "
                f"blocks={score_details['nonempty_blocks']}, "
                f"tables={score_details['table_count']}"
            ),
            "quality_score": score_details["quality_score"],
            "score_breakdown": score_details,
            "page_count": page_count,
            "table_count": table_count,
            "heading_count": heading_count,
            "likely_scanned_pdf": score_details.get("likely_scanned", False),
            "ocr_recommended": score_details.get("ocr_recommended", False),
            "source_format": "pdf",
            "source_file": str(path),
            "errors": all_errors,
        }

        return ParsedDocument(
            title=path.stem,
            content=content,
            blocks=cleaned_blocks,
            source_path=str(path),
            file_type=self.file_type,
            metadata=metadata,
        )
