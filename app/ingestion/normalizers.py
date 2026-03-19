import re
from typing import Any

_PAGE_NOISE_PATTERNS = [
    re.compile(r"^\s*第?\s*\d+\s*页\s*$"),
    re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
]

_TRANSITION_LINES = {
    "例如：",
    "比如：",
    "也就是说：",
    "也就是：",
    "原因是：",
    "可以拆成：",
    "如下：",
    "比如",
    "例如",
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise_line(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True

    for pattern in _PAGE_NOISE_PATTERNS:
        if pattern.match(s):
            return True

    if re.fullmatch(r"[-_=*·•]{3,}", s):
        return True

    return False


def normalize_block_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def normalize_block_type(block_type: str | None) -> str:
    bt = (block_type or "").strip().lower()
    mapping = {
        "title": "heading",
        "heading": "heading",
        "header": "heading",
        "subtitle": "heading",
        "narrativetext": "paragraph",
        "paragraph": "paragraph",
        "text": "paragraph",
        "listitem": "list_item",
        "list_item": "list_item",
        "bullet": "list_item",
        "table": "table",
        "quote": "quote",
        "blockquote": "quote",
        "code": "code",
        "caption": "image_caption",
        "image_caption": "image_caption",
        "metadata": "metadata",
    }
    return mapping.get(bt, "paragraph")


def merge_small_blocks(blocks: list[dict[str, Any]], min_chars: int = 35) -> list[dict[str, Any]]:
    if not blocks:
        return []

    merged: list[dict[str, Any]] = []
    i = 0

    while i < len(blocks):
        cur = dict(blocks[i])
        cur_text = cur.get("text", "").strip()
        cur_type = cur.get("type", "paragraph")
        cur_section = cur.get("section_path")

        should_merge_forward = (
            (
                len(cur_text) < min_chars
                and cur_type in {"paragraph", "list_item", "quote"}
                and cur_text in _TRANSITION_LINES
            )
            or len(cur_text) < 12
        )

        if should_merge_forward and i + 1 < len(blocks):
            nxt = dict(blocks[i + 1])
            nxt_type = nxt.get("type", "paragraph")
            nxt_section = nxt.get("section_path")
            if nxt_type in {"paragraph", "list_item", "quote"} and nxt_section == cur_section:
                nxt["text"] = f"{cur_text}\n{nxt.get('text', '').strip()}".strip()
                blocks[i + 1] = nxt
                i += 1
                continue

        if merged:
            prev = merged[-1]
            same_section = prev.get("section_path") == cur_section
            merge_to_prev = (
                len(cur_text) < min_chars
                and cur_type == prev.get("type")
                and cur_type in {"paragraph", "list_item"}
                and same_section
            )
            if merge_to_prev:
                prev["text"] = f"{prev.get('text', '').strip()}\n{cur_text}".strip()
                i += 1
                continue

        merged.append(cur)
        i += 1

    return merged


def clean_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []

    for idx, block in enumerate(blocks):
        text = normalize_block_text(block.get("text", ""))
        if not text:
            continue
        if is_noise_line(text):
            continue

        item = dict(block)
        item["text"] = text
        item["type"] = normalize_block_type(block.get("type"))
        item["block_index"] = block.get("block_index", idx)
        cleaned.append(item)

    return merge_small_blocks(cleaned)