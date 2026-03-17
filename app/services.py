import math
import os
from pathlib import Path

import requests

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBED_MODEL
from app.db import (
    insert_document,
    get_document_by_id,
    clear_chunks_by_document_id,
    insert_chunk,
    get_chunks_by_document_id,
    get_all_chunks,
)
from app.utils import scan_files, read_text_file



def import_documents(folder: str):
    files = scan_files(folder)
    imported_doc_ids = []

    for file in files:
        content = read_text_file(file)
        title = Path(file).stem
        doc_id = insert_document(title=title, content=content, file_path=file)
        if doc_id:
            imported_doc_ids.append(doc_id)

    return imported_doc_ids


def summarize_text(text: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一个擅长总结文档的助手，请用中文简洁总结。",
            },
            {
                "role": "user",
                "content": f"请总结下面内容，控制在 3 到 5 句话：\n\n{text}",
            },
        ],
        "stream": False,
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


def split_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    最小版字符切分：
    - chunk_size: 每块最大字符数
    - overlap: 相邻块重叠字符数
    """
    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    chunks = []
    start = 0
    text = text.strip()
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "chunk_text": chunk_text,
                    "char_start": start,
                    "char_end": end,
                }
            )

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def get_embedding(text: str) -> list[float]:
    """
    使用 Ollama Embeddings。
    需要本地已有 embedding 模型，例如：
    ollama pull nomic-embed-text
    """
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "prompt": text,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError("Embedding 返回为空")

        return embedding
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Embedding 请求失败: {e}")


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def index_document(
    doc_id: int,
    chunk_size: int = 500,
    overlap: int = 100,
):
    row = get_document_by_id(doc_id)
    if not row:
        raise ValueError("document not found")

    content = row["content"]
    title = row["title"]

    clear_chunks_by_document_id(doc_id)

    chunks = split_text(content, chunk_size=chunk_size, overlap=overlap)
    saved_chunks = []

    for idx, item in enumerate(chunks):
        embedding = get_embedding(item["chunk_text"])
        insert_chunk(
            document_id=doc_id,
            chunk_index=idx,
            chunk_text=item["chunk_text"],
            embedding=embedding,
            char_start=item["char_start"],
            char_end=item["char_end"],
        )

        saved_chunks.append(
            {
                "document_id": doc_id,
                "title": title,
                "chunk_index": idx,
                "char_start": item["char_start"],
                "char_end": item["char_end"],
                "text_preview": item["chunk_text"][:120],
            }
        )

    return {
        "document_id": doc_id,
        "title": title,
        "chunk_count": len(saved_chunks),
        "chunks": saved_chunks,
    }


def retrieve_chunks(query: str, top_k: int = 3):
    query_embedding = get_embedding(query)
    all_chunks = get_all_chunks()

    scored_results = []

    for row in all_chunks:
        try:
            import json

            chunk_embedding = json.loads(row["embedding"])
        except Exception:
            continue

        score = cosine_similarity(query_embedding, chunk_embedding)

        scored_results.append(
            {
                "chunk_id": row["id"],
                "document_id": row["document_id"],
                "title": row["title"],
                "chunk_index": row["chunk_index"],
                "text": row["chunk_text"],
                "score": round(score, 6),
                "char_start": row["char_start"],
                "char_end": row["char_end"],
            }
        )

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:top_k]


def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[来源{i}] 文档: {chunk['title']} | chunk: {chunk['chunk_index']}\n{chunk['text']}"
        )

    context_text = "\n\n".join(context_parts)

    return f"""
你是一个知识库问答助手，请严格根据提供的资料回答问题。

要求：
1. 只能依据“资料片段”作答，不要编造。
2. 如果资料不足以回答，请明确说“根据当前知识库内容无法确定”。
3. 回答使用中文。
4. 回答后附上你引用的来源编号，例如：[来源1]。
5. 尽量简洁清晰。

用户问题：
{question}

资料片段：
{context_text}
""".strip()


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    prompt = build_rag_prompt(question, retrieved_chunks)

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严谨的 RAG 问答助手。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"LLM 请求失败: {e}")
    except KeyError:
        raise RuntimeError("LLM 返回格式异常")


def answer_question(query: str, top_k: int = 3):
    retrieved = retrieve_chunks(query=query, top_k=top_k)

    if not retrieved:
        return {
            "question": query,
            "answer": "当前知识库中没有可用的检索结果。",
            "sources": [],
        }

    answer = generate_answer(question=query, retrieved_chunks=retrieved)

    return {
        "question": query,
        "answer": answer,
        "sources": retrieved,
    }


def get_document_chunks(doc_id: int):
    rows = get_chunks_by_document_id(doc_id)

    results = []
    for row in rows:
        results.append(
            {
                "chunk_id": row["id"],
                "document_id": row["document_id"],
                "title": row["title"],
                "chunk_index": row["chunk_index"],
                "text": row["chunk_text"],
                "char_start": row["char_start"],
                "char_end": row["char_end"],
            }
        )

    return results