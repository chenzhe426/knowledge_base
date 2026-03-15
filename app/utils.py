from pathlib import Path


def read_text_file(file_path: str) -> str:
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def scan_files(folder: str) -> list[str]:
    path = Path(folder)
    files = []

    for file in path.iterdir():
        if file.is_file() and file.suffix in [".txt", ".md"]:
            files.append(str(file))

    return files

def format_document(row):
    return {
        "id": row[0],
        "title": row[1],
        "content": row[2] if len(row) > 2 else None,
        "file_path": row[3] if len(row) > 3 else None,
        "created_at": row[4] if len(row) > 4 else None,
    }