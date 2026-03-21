import json
from typing import Any

import requests

import app.config as config
from app.services.common import normalize_whitespace


def _cfg(name: str, default: Any):
    return getattr(config, name, default)


OLLAMA_BASE_URL = _cfg("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = _cfg("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL = _cfg("OLLAMA_EMBED_MODEL", "nomic-embed-text")

REQUEST_TIMEOUT = int(_cfg("OLLAMA_TIMEOUT", 120))


def _post_json(url: str, payload: dict[str, Any], timeout: int = REQUEST_TIMEOUT) -> dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("invalid json response")
    return data


def get_embedding(text: str) -> list[float]:
    text = normalize_whitespace(text or "")
    if not text:
        return []

    # 优先兼容 /api/embeddings
    try:
        data = _post_json(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/embeddings",
            {
                "model": OLLAMA_EMBED_MODEL,
                "prompt": text,
            },
        )
        embedding = data.get("embedding")
        if isinstance(embedding, list):
            return embedding
    except Exception:
        pass

    # 回退兼容 /api/embed
    try:
        data = _post_json(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed",
            {
                "model": OLLAMA_EMBED_MODEL,
                "input": text,
            },
        )
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            first = embeddings[0]
            if isinstance(first, list):
                return first
        embedding = data.get("embedding")
        if isinstance(embedding, list):
            return embedding
    except Exception:
        pass

    return []


def chat_completion(system_prompt: str, user_prompt: str) -> str:
    system_prompt = normalize_whitespace(system_prompt or "")
    user_prompt = normalize_whitespace(user_prompt or "")

    if not user_prompt:
        return ""

    # 优先兼容 /api/chat
    try:
        data = _post_json(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            {
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )

        message = data.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(data.get("response"), str) and data["response"].strip():
            return data["response"].strip()
    except Exception:
        pass

    # 回退兼容 /api/generate
    try:
        prompt = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
        data = _post_json(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
        )
        response = data.get("response")
        if isinstance(response, str):
            return response.strip()
    except Exception as e:
        raise RuntimeError(f"llm request failed: {e}") from e

    return ""


def summarize_text(text: str) -> str:
    text = normalize_whitespace(text or "")
    if not text:
        return "没有可摘要的内容。"

    system_prompt = (
        "你是一个擅长阅读文档并生成摘要的助手。"
        "请基于给定内容输出中文摘要，尽量准确、简洁、结构清晰。"
    )
    user_prompt = (
        "请总结下面这段内容，输出：\n"
        "1. 一段总体摘要\n"
        "2. 3到5个关键点\n\n"
        f"内容：\n{text}"
    )

    return chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)


def chat_completion(system_prompt: str, user_prompt: str) -> str:
    system_text = normalize_whitespace(system_prompt or "")
    user_text = normalize_whitespace(user_prompt or "")

    messages: list[dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    if user_text:
        messages.append({"role": "user", "content": user_text})

    if not messages:
        return ""

    try:
        data = _post_json(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            {
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": messages,
            },
        )

        message = data.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

        response = data.get("response")
        if isinstance(response, str) and response.strip():
            return response.strip()

        return ""

    except Exception as e:
        raise RuntimeError(f"llm request failed: {e}") from e