from pathlib import Path


def load_text_file(file_path: str | Path) -> str:
    path = Path(file_path)
    return path.read_text(encoding="utf-8", errors="ignore")