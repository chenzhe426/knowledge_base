from __future__ import annotations

import json
from typing import Any, Dict, Generator, Iterable, List

from app.agent.agent import create_kb_agent


def _extract_final_text(result: Dict[str, Any]) -> str:
    messages = result.get("messages", []) or []
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(msg, dict):
            c = msg.get("content")
            if isinstance(c, str) and c.strip():
                return c
    return ""


def _serialize_message(msg: Any) -> Dict[str, Any]:
    if isinstance(msg, dict):
        return {
            "type": msg.get("type") or msg.get("role") or msg.__class__.__name__,
            "content": msg.get("content", ""),
        }

    tool_calls = getattr(msg, "tool_calls", None)
    name = msg.__class__.__name__
    content = getattr(msg, "content", "")

    payload = {
        "type": name,
        "content": content,
    }
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def agent_ask(question: str) -> Dict[str, Any]:
    agent = create_kb_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    messages = result.get("messages", []) or []
    return {
        "ok": True,
        "question": question,
        "answer": _extract_final_text(result),
        "messages": [_serialize_message(m) for m in messages],
    }


def _sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def agent_ask_stream(question: str) -> Generator[str, None, None]:
    agent = create_kb_agent()

    yield _sse({"type": "start", "question": question})

    final_result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    messages = final_result.get("messages", []) or []

    for msg in messages:
        payload = _serialize_message(msg)
        msg_type = payload.get("type", "")

        if "HumanMessage" in msg_type:
            continue

        if payload.get("tool_calls"):
            yield _sse(
                {
                    "type": "tool_call",
                    "message": payload,
                }
            )
        elif "ToolMessage" in msg_type:
            yield _sse(
                {
                    "type": "tool_result",
                    "message": payload,
                }
            )
        else:
            yield _sse(
                {
                    "type": "message",
                    "message": payload,
                }
            )

    yield _sse(
        {
            "type": "final",
            "answer": _extract_final_text(final_result),
        }
    )
    yield _sse({"type": "done"})