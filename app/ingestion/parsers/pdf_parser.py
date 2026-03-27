"""
PDF parser with multi-strategy selection and unified block output.

Fast path:
1. pypdfium2 fast text extraction -> paragraph blocks
2. if quality is good enough, return immediately

Fallback path:
3. pdfplumber -> more detailed extraction, slower
4. fitz raw-text fallback

Design goal:
- no external model dependency
- fast import for digital PDFs
- preserve block structure for downstream chunking
"""

from __future__ import annotations

import logging
import re
import traceback
from pathlib import Path
from typing import Any

from app.ingestion.config import CleaningConfig, PdfParserConfig, ParsingConfig
from app.ingestion.normalizers import blocks_to_content, clean_blocks, normalize_text
from app.ingestion.parsers.base import BaseParser
from app.ingestion.quality import score_pdf_blocks
from app.ingestion.schemas import ParsedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section_path_push(section_stack: list[str], heading_text: str, level: int) -> str:
    while len(section_stack) >= level:
        section_stack.pop()
    section_stack.append(heading_text)
    return " > ".join(section_stack)


def _looks_like_heading(text: str) -> tuple[bool, int | None]:
    """
    Lightweight heading heuristic for fast text parsing.
    Returns (is_heading, heading_level).
    """
    s = text.strip()
    if not s:
        return False, None

    if len(s) > 120:
        return False, None

    # Markdown-ish or numbered headings
    if re.match(r"^(item|section|part)\s+[A-Z0-9IVX.\-]+", s, flags=re.I):
        return True, 2

    if re.match(r"^\d+(\.\d+){0,3}\s+\S+", s):
        return True, 3

    # Mostly uppercase short line
    alpha_chars = [c for c in s if c.isalpha()]
    if alpha_chars:
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if upper_ratio > 0.8 and len(s) < 100:
            return True, 2

    return False, None


def _split_text_to_paragraphs(text: str, max_chars: int = 1200) -> list[str]:
    """
    Split page text into paragraph-like blocks.

    Priority:
    1. blank lines
    2. line-grouping
    3. soft length chunking
    """
    text = normalize_text(text).strip()
    if not text:
        return []

    # First try paragraph split by blank lines
    if "\n\n" in text:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    else:
        # Group short lines into paragraph-like units
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        paras = []
        buf: list[str] = []

        def flush_buf():
            nonlocal buf
            if buf:
                joined = " ".join(buf).strip()
                if joined:
                    paras.append(joined)
            buf = []

        for line in lines:
            # Heading-like lines become standalone paragraphs
            is_heading, _ = _looks_like_heading(line)
            if is_heading:
                flush_buf()
                paras.append(line)
                continue

            buf.append(line)

            # End paragraph if line ends with strong sentence punctuation
            if re.search(r"[.!?:;”\"]\s*$", line):
                flush_buf()

        flush_buf()

    # Soft split overlong paragraphs
    out: list[str] = []
    for para in paras:
        para = para.strip()
        if not para:
            continue

        if len(para) <= max_chars:
            out.append(para)
            continue

        start = 0
        while start < len(para):
            end = start + max_chars
            if end >= len(para):
                chunk = para[start:].strip()
                if chunk:
                    out.append(chunk)
                break

            # Prefer sentence boundary
            split_pt = max(
                para.rfind(". ", start, end),
                para.rfind("; ", start, end),
                para.rfind(": ", start, end),
            )
            if split_pt <= start:
                split_pt = para.rfind(" ", start, end)
            if split_pt <= start:
                split_pt = end

            chunk = para[start:split_pt + 1].strip()
            if chunk:
                out.append(chunk)
            start = split_pt + 1

    return out


def _analyze_pdf_file(path: Path) -> tuple[bool, int, int]:
    """
    Returns:
        (likely_scanned, page_count, file_size_bytes)
    """
    file_size = path.stat().st_size if path.exists() else 0
    page_count = 0

    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        page_count = len(pdf)

        if page_count <= 0:
            return False, 0, file_size

        avg_bytes_per_page = file_size / page_count
        likely_by_size = avg_bytes_per_page > 500 * 1024

        has_any_text = False
        for page_idx in range(min(3, page_count)):
            page = pdf[page_idx]
            textpage = page.get_textpage()
            text = textpage.get_text_bounded()
            if text and text.strip():
                has_any_text = True
                break

        likely_scanned = likely_by_size and not has_any_text
        return likely_scanned, page_count, file_size

    except Exception as e:
        logger.exception("[PdfParser] _analyze_pdf_file failed for %s: %s", path, e)
        return False, 0, file_size


# ---------------------------------------------------------------------------
# Strategy 1 – pypdfium2 fast text -> paragraph blocks
# ---------------------------------------------------------------------------


def _parse_with_pypdfium2_fasttext(path: Path, config: PdfParserConfig) -> list[dict[str, Any]]:
    """
    Fast path:
    - extract full page text using pypdfium2
    - split into paragraph-like blocks
    - lightweight heading detection
    """
    import pypdfium2 as pdfium

    logger.info("[PdfParser] trying parser=pypdfium2_fasttext for %s", path)

    pdf = pdfium.PdfDocument(str(path))
    page_count = len(pdf)

    logger.info("[PdfParser][pypdfium2_fasttext] opened %s with %d pages", path, page_count)

    blocks: list[dict[str, Any]] = []
    section_stack: list[str] = []

    for page_idx in range(page_count):
        page = pdf[page_idx]
        page_width, page_height = page.get_size()

        try:
            textpage = page.get_textpage()
            raw_text = textpage.get_text_bounded()
        except Exception as e:
            logger.warning(
                "[PdfParser][pypdfium2_fasttext] text extraction failed on page=%d file=%s: %s",
                page_idx + 1,
                path,
                e,
            )
            continue

        if not raw_text or len(raw_text.strip()) < 10:
            logger.debug(
                "[PdfParser][pypdfium2_fasttext] empty/short text on page=%d file=%s",
                page_idx + 1,
                path,
            )
            continue

        paragraphs = _split_text_to_paragraphs(raw_text, max_chars=1200)

        for para in paragraphs:
            para = para.strip()
            if len(para) < 20:
                continue

            is_heading, heading_level = _looks_like_heading(para)
            block_type = "heading" if is_heading else "paragraph"

            section_path = None
            if block_type == "heading" and heading_level is not None:
                section_path = _section_path_push(section_stack, para, heading_level)
            elif section_stack:
                section_path = " > ".join(section_stack)

            blocks.append(
                {
                    "type": block_type,
                    "text": para,
                    "page": page_idx + 1,
                    "heading_level": heading_level,
                    "section_path": section_path,
                    "parser_source": "pypdfium2_fasttext",
                    "bbox": (0.0, 0.0, round(page_width, 1), round(page_height, 1)),
                }
            )

    logger.info(
        "[PdfParser][pypdfium2_fasttext] extracted %d blocks from %s",
        len(blocks),
        path,
    )
    return blocks


# ---------------------------------------------------------------------------
# Strategy 2 – pdfplumber fallback
# ---------------------------------------------------------------------------


def _parse_with_pdfplumber(path: Path, config: PdfParserConfig) -> list[dict[str, Any]]:
    """
    Slower but more detailed fallback parser.
    """
    import pdfplumber

    logger.info("[PdfParser] trying parser=pdfplumber for %s", path)

    blocks: list[dict[str, Any]] = []
    section_stack: list[str] = []

    with pdfplumber.open(str(path)) as pdf:
        logger.info("[PdfParser][pdfplumber] opened %s with %d pages", path, len(pdf.pages))

        for page_idx, page in enumerate(pdf.pages, start=1):
            page_height = page.height

            # Optional tables
            if getattr(config, "extract_tables", False):
                try:
                    tables = page.extract_tables()
                except Exception as e:
                    logger.warning(
                        "[PdfParser][pdfplumber] table extraction failed on page=%d file=%s: %s",
                        page_idx,
                        path,
                        e,
                    )
                    tables = []

                for table_rows in tables or []:
                    if not table_rows:
                        continue
                    table_lines: list[str] = []
                    for row in table_rows:
                        if not row:
                            continue
                        cells = [str(c).strip() if c is not None else "" for c in row]
                        table_lines.append(" | ".join(cells))
                    if table_lines:
                        blocks.append(
                            {
                                "type": "table",
                                "text": "\n".join(table_lines),
                                "page": page_idx,
                                "section_path": None,
                                "parser_source": "pdfplumber",
                                "bbox": None,
                            }
                        )

            try:
                words = page.extract_words()
            except Exception as e:
                logger.warning(
                    "[PdfParser][pdfplumber] extract_words failed on page=%d file=%s: %s",
                    page_idx,
                    path,
                    e,
                )
                continue

            if not words:
                continue

            lines_by_y: dict[float, list[dict[str, Any]]] = {}
            for w in words:
                top = round(w["top"], 1)
                lines_by_y.setdefault(top, []).append(w)

            for y_pos, line_words in sorted(lines_by_y.items()):
                line_words.sort(key=lambda w: w["x0"])
                line_text = " ".join(w["text"] for w in line_words).strip()
                if not line_text:
                    continue

                btype = "paragraph"
                heading_level = None
                section_path = None

                if getattr(config, "remove_header_footer_by_coords", False):
                    dist_from_top = y_pos
                    dist_from_bottom = page_height - y_pos
                    if (
                        dist_from_top < config.header_footer_y_threshold
                        or dist_from_bottom < config.header_footer_y_threshold
                    ):
                        btype = "header" if dist_from_top < config.header_footer_y_threshold else "footer"

                if btype == "paragraph":
                    is_heading, heading_level = _looks_like_heading(line_text)
                    if is_heading:
                        btype = "heading"
                        heading_level = heading_level or 3
                        section_path = _section_path_push(section_stack, line_text, heading_level)
                    elif section_stack:
                        section_path = " > ".join(section_stack)

                x0 = min(w["x0"] for w in line_words)
                x1 = max(w["x1"] for w in line_words)
                y1 = max(w["bottom"] for w in line_words)
                bbox = (round(x0, 1), round(y_pos, 1), round(x1, 1), round(y1, 1))

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

    logger.info("[PdfParser][pdfplumber] extracted %d blocks from %s", len(blocks), path)
    return blocks


# ---------------------------------------------------------------------------
# Final fallback – fitz raw text
# ---------------------------------------------------------------------------


def _parse_with_fitz_rawtext(path: Path) -> list[dict[str, Any]]:
    import fitz

    logger.info("[PdfParser] trying parser=fitz_rawtext for %s", path)

    blocks: list[dict[str, Any]] = []

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        logger.exception("[PdfParser][fitz_rawtext] fitz open failed for %s: %s", path, e)
        return []

    for page_idx, page in enumerate(doc, start=1):
        try:
            raw = page.get_text("text")
        except Exception as e:
            logger.warning(
                "[PdfParser][fitz_rawtext] get_text failed on page=%d file=%s: %s",
                page_idx,
                path,
                e,
            )
            continue

        if not raw or len(raw.strip()) < 10:
            continue

        paragraphs = _split_text_to_paragraphs(raw, max_chars=1200)

        for para in paragraphs:
            if len(para) < 20:
                continue
            is_heading, heading_level = _looks_like_heading(para)
            blocks.append(
                {
                    "type": "heading" if is_heading else "paragraph",
                    "text": para,
                    "page": page_idx,
                    "heading_level": heading_level,
                    "section_path": None,
                    "parser_source": "fitz_rawtext",
                    "bbox": None,
                }
            )

    doc.close()
    logger.info("[PdfParser][fitz_rawtext] extracted %d blocks from %s", len(blocks), path)
    return blocks


# ---------------------------------------------------------------------------
# Candidate runner
# ---------------------------------------------------------------------------


def _run_pdf_candidate(
    parser_name: str,
    parser_func,
    path: Path,
    config: PdfParserConfig,
) -> dict[str, Any] | None:
    try:
        blocks = parser_func(path, config)
        if not blocks:
            logger.warning("[PdfParser] parser=%s returned 0 blocks for %s", parser_name, path)
            return {
                "parser_name": parser_name,
                "blocks": [],
                "quality_score": 0.0,
                "score_details": {},
                "errors": [f"{parser_name}: returned 0 blocks"],
            }

        score_result = score_pdf_blocks(blocks, config)
        logger.info(
            "[PdfParser] parser=%s succeeded for %s | score=%.4f blocks=%d chars=%s tables=%s",
            parser_name,
            path,
            score_result.get("quality_score", 0.0),
            score_result.get("nonempty_blocks", len(blocks)),
            score_result.get("total_chars"),
            score_result.get("table_count"),
        )
        return {
            "parser_name": parser_name,
            "blocks": blocks,
            "quality_score": score_result["quality_score"],
            "score_details": score_result,
            "errors": [],
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(
            "[PdfParser] parser=%s failed for %s: %s\n%s",
            parser_name,
            path,
            e,
            tb,
        )
        return {
            "parser_name": parser_name,
            "blocks": [],
            "quality_score": 0.0,
            "score_details": {},
            "errors": [f"{parser_name}: {e}", tb],
        }


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------


class PdfParser(BaseParser):
    """
    Strategy order:
    1. pypdfium2_fasttext (fast, paragraph blocks)
    2. pdfplumber (slow fallback)
    3. fitz_rawtext (last fallback)

    Important:
    - fast path returns early if score is good enough
    - downstream chunker can still merge blocks normally
    """

    file_type = "pdf"

    def __init__(self, config: ParsingConfig | None = None):
        self._cfg = config.pdf if config else PdfParserConfig()
        self._clean_cfg = config.cleaning if config else CleaningConfig()

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        logger.info("[PdfParser] starting parse for %s", path)

        likely_scanned, analyzed_page_count, file_size = _analyze_pdf_file(path)
        logger.info(
            "[PdfParser] file analysis | path=%s likely_scanned=%s pages=%d size_mb=%.2f",
            path,
            likely_scanned,
            analyzed_page_count,
            file_size / 1024 / 1024 if file_size else 0.0,
        )

        candidates: dict[str, Any] = {}
        all_errors: list[str] = []

        # Fast path first
        fast_result = _run_pdf_candidate(
            "pypdfium2_fasttext",
            _parse_with_pypdfium2_fasttext,
            path,
            self._cfg,
        )
        if fast_result:
            candidates["pypdfium2_fasttext"] = fast_result
            if fast_result.get("errors"):
                all_errors.extend(fast_result["errors"])

            if fast_result.get("blocks") and fast_result["quality_score"] >= self._cfg.min_quality_score:
                logger.info(
                    "[PdfParser] early return with parser=pypdfium2_fasttext for %s score=%.4f",
                    path,
                    fast_result["quality_score"],
                )
                return self._build_parsed_document(
                    path=path,
                    blocks=fast_result["blocks"],
                    parser_used="pypdfium2_fasttext",
                    score_details=fast_result["score_details"],
                    candidate_scores={k: v["quality_score"] for k, v in candidates.items()},
                    file_size=file_size,
                    likely_scanned=likely_scanned,
                    page_count_fallback=analyzed_page_count,
                    errors=all_errors,
                )

        # Slow fallback
        plumber_result = _run_pdf_candidate(
            "pdfplumber",
            _parse_with_pdfplumber,
            path,
            self._cfg,
        )
        if plumber_result:
            candidates["pdfplumber"] = plumber_result
            if plumber_result.get("errors"):
                all_errors.extend(plumber_result["errors"])

        valid_candidates = {k: v for k, v in candidates.items() if v.get("blocks")}

        if valid_candidates:
            best_name = max(valid_candidates, key=lambda k: valid_candidates[k]["quality_score"])
            best = valid_candidates[best_name]
            logger.info(
                "[PdfParser] selected parser=%s for %s with score=%.4f",
                best_name,
                path,
                best["quality_score"],
            )
            return self._build_parsed_document(
                path=path,
                blocks=best["blocks"],
                parser_used=best_name,
                score_details=best["score_details"],
                candidate_scores={k: v["quality_score"] for k, v in valid_candidates.items()},
                file_size=file_size,
                likely_scanned=likely_scanned,
                page_count_fallback=analyzed_page_count,
                errors=all_errors,
            )

        # Last fallback
        logger.warning(
            "[PdfParser] all primary parsers failed for %s; trying fitz raw-text fallback",
            path,
        )

        try:
            fallback_blocks = _parse_with_fitz_rawtext(path)
        except Exception as e:
            tb = traceback.format_exc()
            all_errors.append(f"fitz_rawtext: {e}")
            all_errors.append(tb)
            fallback_blocks = []

        if fallback_blocks:
            fallback_score = score_pdf_blocks(fallback_blocks, self._cfg)
            return self._build_parsed_document(
                path=path,
                blocks=fallback_blocks,
                parser_used="fitz_rawtext_fallback",
                score_details=fallback_score,
                candidate_scores={"fitz_rawtext_fallback": fallback_score["quality_score"]},
                file_size=file_size,
                likely_scanned=likely_scanned,
                page_count_fallback=analyzed_page_count,
                errors=all_errors,
                fallback_used=True,
            )

        if likely_scanned:
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
                    "page_count": analyzed_page_count,
                    "file_size_bytes": file_size,
                    "table_count": 0,
                    "heading_count": 0,
                    "likely_scanned_pdf": True,
                    "ocr_recommended": True,
                    "source_format": "pdf",
                    "source_file": str(path),
                    "errors": all_errors,
                    "hint": "This PDF appears to be scanned/image-based. Add OCR before import.",
                },
            )

        raise RuntimeError(
            f"all PDF parsers failed for {path}. "
            f"likely_scanned={likely_scanned}, pages={analyzed_page_count}, "
            f"size_bytes={file_size}. errors={all_errors}"
        )

    def _build_parsed_document(
        self,
        path: Path,
        blocks: list[dict[str, Any]],
        parser_used: str,
        score_details: dict[str, Any],
        candidate_scores: dict[str, float],
        file_size: int,
        likely_scanned: bool,
        page_count_fallback: int,
        errors: list[str],
        fallback_used: bool = False,
    ) -> ParsedDocument:
        for block in blocks:
            block["source_format"] = "pdf"
            block.setdefault("page", None)

        cleaned_blocks = clean_blocks(blocks, self._clean_cfg)
        for b in cleaned_blocks:
            b.setdefault("source_format", "pdf")

        content = blocks_to_content(
            cleaned_blocks,
            include_headings=self._clean_cfg.include_headings_in_content,
        )

        table_count = sum(1 for b in cleaned_blocks if b.get("type") == "table")
        heading_count = sum(1 for b in cleaned_blocks if b.get("type") == "heading")
        page_count = len({b.get("page") for b in cleaned_blocks if b.get("page") is not None})
        if page_count == 0:
            page_count = page_count_fallback

        metadata = {
            "parser_used": parser_used,
            "candidate_scores": candidate_scores,
            "selection_reason": (
                f"score={score_details.get('quality_score', 0.0):.3f} vs "
                f"min={self._cfg.min_quality_score}; "
                f"chars={score_details.get('total_chars')}, "
                f"blocks={score_details.get('nonempty_blocks')}, "
                f"tables={score_details.get('table_count')}"
            ),
            "quality_score": score_details.get("quality_score", 0.0),
            "score_breakdown": score_details,
            "page_count": page_count,
            "table_count": table_count,
            "heading_count": heading_count,
            "likely_scanned_pdf": likely_scanned,
            "ocr_recommended": likely_scanned,
            "source_format": "pdf",
            "source_file": str(path),
            "file_size_bytes": file_size,
            "errors": errors,
            "fallback_used": fallback_used,
        }

        return ParsedDocument(
            title=path.stem,
            content=content,
            blocks=cleaned_blocks,
            source_path=str(path),
            file_type=self.file_type,
            metadata=metadata,
        )