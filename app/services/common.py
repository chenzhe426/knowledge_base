import json
import re
from typing import Any


def safe_json_loads(value: Any, default: Any):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def truncate(text: str, max_len: int = 200) -> str:
    text = normalize_whitespace(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def safe_get(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def normalize_section_path(section_path: Any) -> list[str]:
    if not section_path:
        return []
    if isinstance(section_path, list):
        return [str(x).strip() for x in section_path if str(x).strip()]
    if isinstance(section_path, str):
        raw = section_path.strip()
        if not raw:
            return []
        if ">" in raw:
            return [x.strip() for x in raw.split(">") if x.strip()]
        return [raw]
    return [str(section_path).strip()]


def section_path_to_str(section_path: Any) -> str:
    parts = normalize_section_path(section_path)
    return " > ".join(parts) if parts else ""


def last_section_title(section_path: Any) -> str:
    parts = normalize_section_path(section_path)
    return parts[-1] if parts else ""


def normalize_embedding(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [to_float(x) for x in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [to_float(x) for x in parsed]
        except Exception:
            return []
    return []