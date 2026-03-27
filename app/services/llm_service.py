import json
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from threading import Lock
from typing import Any, Optional

import requests

import app.config as config
from app.services.common import normalize_whitespace


def _cfg(name: str, default: Any):
    return getattr(config, name, default)


OLLAMA_BASE_URL = config.OLLAMA_BASE_URL
OLLAMA_MODEL = config.OLLAMA_MODEL
OLLAMA_EMBED_MODEL = config.OLLAMA_EMBED_MODEL
REQUEST_TIMEOUT = int(config.OLLAMA_TIMEOUT)


# =============================================================================
# Bounded LRU Cache for embeddings (avoids unbounded memory growth)
# =============================================================================

_EMBEDDING_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_EMBEDDING_CACHE_LOCK = Lock()
_EMBEDDING_CACHE_MAX_SIZE = 2048


def _get_cached_embedding(key: str) -> Optional[list[float]]:
    with _EMBEDDING_CACHE_LOCK:
        if key in _EMBEDDING_CACHE:
            _EMBEDDING_CACHE.move_to_end(key)
            return _EMBEDDING_CACHE[key]
    return None


def _set_cached_embedding(key: str, value: list[float]) -> None:
    with _EMBEDDING_CACHE_LOCK:
        if key in _EMBEDDING_CACHE:
            _EMBEDDING_CACHE.move_to_end(key)
        else:
            if len(_EMBEDDING_CACHE) >= _EMBEDDING_CACHE_MAX_SIZE:
                _EMBEDDING_CACHE.popitem(last=False)
        _EMBEDDING_CACHE[key] = value


def _embedding_cache_key(text: str) -> str:
    return text[:512]


@lru_cache(maxsize=1024)
def get_embedding(text: str) -> list[float]:
    text = normalize_whitespace(text or "")
    if not text:
        return []

    cache_key = _embedding_cache_key(text)
    cached = _get_cached_embedding(cache_key)
    if cached is not None:
        return cached

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
            _set_cached_embedding(cache_key, embedding)
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
                _set_cached_embedding(cache_key, first)
                return first

        embedding = data.get("embedding")
        if isinstance(embedding, list):
            _set_cached_embedding(cache_key, embedding)
            return embedding
    except Exception:
        pass

    return []


def get_embeddings_batch(texts: list[str], max_workers: int = 8) -> list[list[float]]:
    """
    并行调用 get_embedding，适用于批量生成 chunk embedding。
    返回顺序与输入顺序一致。
    """
    if not texts:
        return []

    results: list[list[float]] = [None] * len(texts)

    def worker(idx: int, text: str) -> tuple[int, list[float]]:
        return idx, get_embedding(text)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, i, t): i for i, t in enumerate(texts)}
        for future in as_completed(futures):
            try:
                idx, embedding = future.result()
                results[idx] = embedding
            except Exception:
                results[idx] = []

    # 兜底：确保所有槽位都有返回值
    return [r if r is not None else [] for r in results]


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


# =============================================================================
# Structured JSON chat completion (for reranker / verifier)
# =============================================================================

def chat_completion_json(
    system_prompt: str,
    user_prompt: str,
    default: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Call chat_completion and try to parse the response as JSON.
    Falls back to `default` if parsing fails or call fails.
    """
    if default is None:
        default = {}

    try:
        raw = chat_completion(system_prompt, user_prompt)
        if not raw:
            return default

        # Try direct JSON parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any {...} in the response
        brace_match = re.search(r"(\{.*\})", raw, re.S)
        if brace_match:
            try:
                return json.loads(brace_match.group(1))
            except json.JSONDecodeError:
                pass

        return default
    except Exception:
        return default


def chat_completion_raw(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """
    Low-level chat completion with more control over parameters.
    Returns raw text. Raises on failure.
    """
    system_text = normalize_whitespace(system_prompt or "")
    user_text = normalize_whitespace(user_prompt or "")

    if not system_text and not user_text:
        return ""

    base_url = OLLAMA_BASE_URL.rstrip("/")
    model_name = model or OLLAMA_MODEL

    messages: list[dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    if user_text:
        messages.append({"role": "user", "content": user_text})

    payload: dict[str, Any] = {
        "model": model_name,
        "stream": False,
        "messages": messages,
    }
    if temperature > 0:
        payload["temperature"] = temperature

    try:
        data = _post_json(f"{base_url}/api/chat", payload)
        content = _extract_chat_content(data)
        if content:
            return content
    except Exception:
        pass

    # Fallback /api/generate
    prompt_parts: list[str] = []
    if system_text:
        prompt_parts.append(f"系统指令：\n{system_text}")
    if user_text:
        prompt_parts.append(f"用户问题：\n{user_text}")
    prompt = "\n\n".join(prompt_parts).strip()

    fallback_payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }
    if temperature > 0:
        fallback_payload["temperature"] = temperature

    data = _post_json(f"{base_url}/api/generate", fallback_payload)
    content = _extract_generate_content(data)
    if content:
        return content

    raise RuntimeError("chat_completion_raw: all backends failed")
