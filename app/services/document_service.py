import inspect
import json
from pathlib import Path
from typing import Any

from app.config import DATA_DIR
from app.db import get_all_documents, get_chunks_by_document_id, insert_document
from app.ingestion.pipeline import parse_document, parse_documents_from_folder
from app.ingestion.schemas import ParsedDocument
from app.services.chunk_service import block_to_dict, build_blocks_from_content
from app.services.common import ensure_list, safe_get, safe_json_loads


def _call_insert_document(payload: dict[str, Any]) -> Any:
    sig = inspect.signature(insert_document)
    accepted = set(sig.parameters.keys())
    kwargs = {k: v for k, v in payload.items() if k in accepted}
    try:
        return insert_document(**kwargs)
    except TypeError:
        common_order = [
            "title",
            "content",
            "source_path",
            "file_type",
            "metadata_json",
            "blocks_json",
        ]
        args = [payload.get(k) for k in common_order if k in payload]
        return insert_document(*args)


def _parsed_document_to_payload(parsed: ParsedDocument | Any) -> dict[str, Any]:
    title = safe_get(parsed, "title", "") or ""
    content = safe_get(parsed, "content", "") or ""
    source_path = safe_get(parsed, "source_path", "") or ""
    file_type = safe_get(parsed, "file_type", "") or Path(source_path).suffix.lstrip(".").lower()
    metadata = safe_get(parsed, "metadata", {}) or {}
    blocks = safe_get(parsed, "blocks", None)

    if not blocks:
        blocks = build_blocks_from_content(content)

    blocks_dict = [block_to_dict(b) for b in ensure_list(blocks)]

    return {
        "title": title or Path(source_path).stem or "Untitled",
        "content": content,
        "source_path": source_path,
        "file_type": file_type,
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
        "blocks_json": json.dumps(blocks_dict, ensure_ascii=False),
        "blocks_count": len(blocks_dict),
        "char_count": len(content or ""),
    }


def import_single_document(file_path: str | Path) -> dict[str, Any]:
    parsed = parse_document(str(file_path))
    payload = _parsed_document_to_payload(parsed)
    inserted = _call_insert_document(payload)

    doc_id = inserted if isinstance(inserted, int) else (
        inserted.get("id") if isinstance(inserted, dict) else None
    )

    return {
        "id": doc_id,
        "title": payload["title"],
        "source_path": payload["source_path"],
        "file_type": payload["file_type"],
        "char_count": payload["char_count"],
        "blocks_count": payload["blocks_count"],
    }


def import_documents(folder: str | Path = DATA_DIR) -> dict[str, Any]:
    parsed_documents = parse_documents_from_folder(str(folder))
    results = []
    success = 0
    failed = 0

    for item in parsed_documents:
        try:
            payload = _parsed_document_to_payload(item)
            inserted = _call_insert_document(payload)
            doc_id = inserted if isinstance(inserted, int) else (
                inserted.get("id") if isinstance(inserted, dict) else None
            )
            results.append(
                {
                    "success": True,
                    "id": doc_id,
                    "title": payload["title"],
                    "source_path": payload["source_path"],
                    "file_type": payload["file_type"],
                    "char_count": payload["char_count"],
                    "blocks_count": payload["blocks_count"],
                }
            )
            success += 1
        except Exception as e:
            failed += 1
            results.append(
                {
                    "success": False,
                    "source_path": safe_get(item, "source_path", ""),
                    "error": str(e),
                }
            )

    return {
        "total": len(parsed_documents),
        "success": success,
        "failed": failed,
        "items": results,
    }


def list_documents() -> list[dict[str, Any]]:
    rows = get_all_documents() or []
    results = []

    for row in rows:
        blocks = safe_json_loads(safe_get(row, "blocks_json"), default=[])
        metadata = safe_json_loads(safe_get(row, "metadata_json"), default={})
        content = safe_get(row, "content", "") or ""

        results.append(
            {
                "id": safe_get(row, "id"),
                "title": safe_get(row, "title", ""),
                "source_path": safe_get(row, "source_path", ""),
                "file_type": safe_get(row, "file_type", ""),
                "char_count": len(content),
                "blocks_count": len(blocks) if isinstance(blocks, list) else 0,
                "metadata": metadata,
            }
        )

    return results


def get_document_chunks(doc_id: int) -> list[dict[str, Any]]:
    rows = get_chunks_by_document_id(doc_id) or []
    results = []

    for row in rows:
        metadata = safe_json_loads(safe_get(row, "metadata_json"), default={})
        results.append(
            {
                "id": safe_get(row, "id"),
                "document_id": safe_get(row, "document_id"),
                "chunk_index": safe_get(row, "chunk_index"),
                "chunk_text": safe_get(row, "chunk_text", ""),
                "chunk_type": safe_get(row, "chunk_type"),
                "section_path": safe_get(row, "section_path", ""),
                "page_start": safe_get(row, "page_start"),
                "page_end": safe_get(row, "page_end"),
                "block_start_index": safe_get(row, "block_start_index"),
                "block_end_index": safe_get(row, "block_end_index"),
                "metadata": metadata,
            }
        )
    return results