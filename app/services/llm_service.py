from typing import Any

import requests

from app.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, OLLAMA_MODEL


def _ollama_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def get_embedding(text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        return []

    try:
        data = _ollama_post(
            "/api/embeddings",
            {"model": OLLAMA_EMBED_MODEL, "prompt": text},
        )
        emb = data.get("embedding") or []
        return [float(x) for x in emb]
    except Exception:
        data = _ollama_post(
            "/api/embed",
            {"model": OLLAMA_EMBED_MODEL, "input": text},
        )
        emb_list = data.get("embeddings") or []
        if emb_list and isinstance(emb_list, list):
            return [float(x) for x in emb_list[0]]
        return []


def chat_completion(prompt: str, system: str | None = None) -> str:
    final_prompt = prompt
    if system:
        final_prompt = f"{system.strip()}\n\n{prompt.strip()}"

    data = _ollama_post(
        "/api/generate",
        {
            "model": OLLAMA_MODEL,
            "prompt": final_prompt,
            "stream": False,
        },
    )
    return (data.get("response") or "").strip()


def summarize_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    prompt = (
        "请用中文对下面内容做简洁摘要，保留核心信息，尽量条理清楚：\n\n"
        f"{text}"
    )
    return chat_completion(prompt)