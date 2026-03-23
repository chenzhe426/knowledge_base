import json
from typing import Any

import requests

import app.config as config
from app.services.common import normalize_whitespace



def _cfg(name: str, default: Any):
    return getattr(config, name, default)


OLLAMA_BASE_URL = config.OLLAMA_BASE_URL
OLLAMA_MODEL = config.OLLAMA_MODEL
OLLAMA_EMBED_MODEL = config.OLLAMA_EMBED_MODEL
REQUEST_TIMEOUT = int(config.OLLAMA_TIMEOUT)


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout: int = REQUEST_TIMEOUT,
) -> dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()

    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("invalid json response")
    return data


def _extract_chat_content(data: dict[str, Any]) -> str:
    """
    兼容 Ollama /api/chat 和部分兼容返回格式
    """
    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    response = data.get("response")
    if isinstance(response, str) and response.strip():
        return response.strip()

    return ""


def _extract_generate_content(data: dict[str, Any]) -> str:
    """
    兼容 Ollama /api/generate 返回格式
    """
    response = data.get("response")
    if isinstance(response, str) and response.strip():
        return response.strip()

    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


def get_embedding(text: str) -> list[float]:
    text = normalize_whitespace(text or "")
    if not text:
        return []

    base_url = OLLAMA_BASE_URL.rstrip("/")

    # 优先兼容 /api/embeddings
    try:
        data = _post_json(
            f"{base_url}/api/embeddings",
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
            f"{base_url}/api/embed",
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
    """
    对外统一接口：
    chat_completion(system_prompt=..., user_prompt=...)
    """
    system_text = normalize_whitespace(system_prompt or "")
    user_text = normalize_whitespace(user_prompt or "")

    if not system_text and not user_text:
        return ""

    base_url = OLLAMA_BASE_URL.rstrip("/")

    messages: list[dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    if user_text:
        messages.append({"role": "user", "content": user_text})

    # 优先走 /api/chat
    try:
        data = _post_json(
            f"{base_url}/api/chat",
            {
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": messages,
            },
        )
        content = _extract_chat_content(data)
        if content:
            return content
    except Exception:
        pass

    # 回退 /api/generate
    try:
        prompt_parts: list[str] = []
        if system_text:
            prompt_parts.append(f"系统指令：\n{system_text}")
        if user_text:
            prompt_parts.append(f"用户问题：\n{user_text}")

        prompt = "\n\n".join(prompt_parts).strip()

        data = _post_json(
            f"{base_url}/api/generate",
            {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
        )
        content = _extract_generate_content(data)
        if content:
            return content

        return ""
    except Exception as e:
        raise RuntimeError(f"llm request failed: {e}") from e