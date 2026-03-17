from pathlib import Path
from app.utils import scan_files, read_text_file
from app.db import insert_document
import requests
import os
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL

def import_documents(folder: str):
    files = scan_files(folder)

    for file in files:
        content = read_text_file(file)
        title = Path(file).stem
        insert_document(title=title, content=content, file_path=file)

def summarize_text(text: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一个擅长总结文档的助手，请用中文简洁总结。"
            },
            {
                "role": "user",
                "content": f"请总结下面内容，控制在 3 到 5 句话：\n\n{text}"
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama 请求失败: {e}")
    except KeyError:
        raise RuntimeError("Ollama 返回格式异常")