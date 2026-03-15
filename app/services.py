from pathlib import Path
from app.utils import scan_files, read_text_file
from app.db import insert_document
import requests
import os

def import_documents(folder: str):
    files = scan_files(folder)

    for file in files:
        content = read_text_file(file)
        title = Path(file).stem
        insert_document(title=title, content=content, file_path=file)

def summarize_text(text: str) -> str:
    api_key = os.getenv("API_KEY")
    url = "https://example.com/summarize"

    payload = {
        "text": text
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()
    return data.get("summary", "")