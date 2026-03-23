import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client, monkeypatch):
    monkeypatch.setattr("app.api.init_db", lambda: None, raising=False)

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_import_file_response_contract(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.import_single_document",
        lambda file_path: {
            "id": 101,
            "title": "员工手册",
            "file_path": file_path,
            "file_type": "pdf",
            "source_type": "upload",
            "lang": "zh",
            "author": "HR",
            "block_count": 12,
            "metadata": {"department": "hr"},
            "tags": ["handbook", "policy"],
        },
    )

    resp = client.post("/import/file", json={"file_path": "/tmp/handbook.pdf"})
    assert resp.status_code == 200

    data = resp.json()
    assert data["ok"] is True
    assert "document" in data

    doc = data["document"]
    assert doc["id"] == 101
    assert doc["title"] == "员工手册"
    assert doc["file_path"] == "/tmp/handbook.pdf"
    assert doc["file_type"] == "pdf"
    assert doc["source_type"] == "upload"
    assert doc["lang"] == "zh"
    assert doc["author"] == "HR"
    assert doc["block_count"] == 12
    assert doc["metadata"] == {"department": "hr"}
    assert doc["tags"] == ["handbook", "policy"]


def test_import_folder_response_contract(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.import_documents",
        lambda folder: [
            {
                "id": 1,
                "title": "文档A",
                "file_path": f"{folder}/a.txt",
                "file_type": "txt",
                "source_type": "folder_import",
                "lang": "zh",
                "author": None,
                "block_count": 3,
                "metadata": {},
                "tags": [],
            },
            {
                "id": 2,
                "title": "文档B",
                "file_path": f"{folder}/b.md",
                "file_type": "md",
                "source_type": "folder_import",
                "lang": "zh",
                "author": None,
                "block_count": 5,
                "metadata": {"team": "ops"},
                "tags": ["kb"],
            },
        ],
    )

    resp = client.post("/import/folder", json={"folder": "/tmp/docs"})
    assert resp.status_code == 200

    data = resp.json()
    assert data["ok"] is True
    assert data["count"] == 2
    assert isinstance(data["documents"], list)
    assert len(data["documents"]) == 2
    assert data["documents"][0]["id"] == 1
    assert data["documents"][1]["title"] == "文档B"


def test_index_requires_document_id(client):
    resp = client.post("/index", json={"chunk_size": 800, "overlap": 100})
    assert resp.status_code == 422


def test_index_accepts_document_id(client, monkeypatch):
    called = {}

    def fake_index_document(document_id: int, chunk_size: int, overlap: int):
        called["document_id"] = document_id
        called["chunk_size"] = chunk_size
        called["overlap"] = overlap
        return {
            "document_id": document_id,
            "title": "制度文档",
            "chunk_count": 2,
            "chunks": [
                {
                    "id": 11,
                    "chunk_index": 0,
                    "chunk_type": "paragraph",
                    "section_path": "第一章 > 总则",
                    "section_title": "总则",
                    "page_start": 1,
                    "page_end": 1,
                    "token_count": 120,
                    "block_start_index": 0,
                    "block_end_index": 1,
                    "preview": "这是第一段预览",
                },
                {
                    "id": 12,
                    "chunk_index": 1,
                    "chunk_type": "paragraph",
                    "section_path": "第二章 > 请假",
                    "section_title": "请假",
                    "page_start": 2,
                    "page_end": 2,
                    "token_count": 98,
                    "block_start_index": 2,
                    "block_end_index": 3,
                    "preview": "这是第二段预览",
                },
            ],
        }

    monkeypatch.setattr("app.api.index_document", fake_index_document)

    resp = client.post("/index", json={"document_id": 9, "chunk_size": 800, "overlap": 100})
    assert resp.status_code == 200

    data = resp.json()
    assert called == {"document_id": 9, "chunk_size": 800, "overlap": 100}
    assert data["ok"] is True
    assert data["document_id"] == 9
    assert data["title"] == "制度文档"
    assert data["chunk_count"] == 2
    assert len(data["chunks"]) == 2
    assert data["chunks"][0]["chunk_index"] == 0
    assert data["chunks"][0]["section_title"] == "总则"


def test_summary_requires_document_id(client):
    resp = client.post("/summary", json={})
    assert resp.status_code == 422


def test_summary_accepts_document_id(client, monkeypatch):
    called = {}

    def fake_summarize_document(document_id: int):
        called["document_id"] = document_id
        return {
            "document_id": document_id,
            "title": "员工守则",
            "summary": "这是一份关于员工守则的摘要。",
        }

    monkeypatch.setattr("app.api.summarize_document", fake_summarize_document)

    resp = client.post("/summary", json={"document_id": 3})
    assert resp.status_code == 200

    data = resp.json()
    assert called["document_id"] == 3
    assert data["ok"] is True
    assert data["document_id"] == 3
    assert data["title"] == "员工守则"
    assert data["summary"] == "这是一份关于员工守则的摘要。"


def test_ask_response_contract(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.answer_question",
        lambda **kwargs: {
            "question": kwargs["question"],
            "rewritten_query": "员工请假制度",
            "answer": "员工请假需要提前提交申请。",
            "structured": None,
            "sources": [
                {
                    "chunk_id": 1,
                    "document_id": 10,
                    "doc_title": "员工手册",
                    "section_title": "请假制度",
                    "section_path": "第二章 > 请假制度",
                    "page_start": 3,
                    "page_end": 3,
                    "quote": "员工请假需要提前提交申请。",
                    "score": 0.91,
                    "highlight_spans": [],
                }
            ],
            "retrieved_chunks": [
                {
                    "chunk_id": 1,
                    "document_id": 10,
                    "chunk_index": 0,
                    "score": 0.91,
                    "embedding_score": 0.88,
                    "keyword_score": 0.93,
                    "bm25_score": 1.2,
                    "title_match_score": 0.4,
                    "section_match_score": 0.6,
                    "coverage_score": 0.7,
                    "matched_term_count": 2,
                    "doc_title": "员工手册",
                    "section_title": "请假制度",
                    "section_path": "第二章 > 请假制度",
                    "page_start": 3,
                    "page_end": 3,
                    "chunk_type": "paragraph",
                    "chunk_text": "员工请假需要提前提交申请。",
                    "term_hits": {"请假": 1, "申请": 1},
                    "term_hit_detail": {},
                    "is_neighbor": False,
                }
            ],
            "confidence": 0.89,
            "session_id": "sess_123",
        },
    )

    resp = client.post(
        "/ask",
        json={
            "question": "请假怎么申请？",
            "top_k": 5,
            "response_mode": "text",
            "highlight": True,
            "session_id": None,
            "use_chat_context": True,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert data["question"] == "请假怎么申请？"
    assert data["rewritten_query"] == "员工请假制度"
    assert data["answer"] == "员工请假需要提前提交申请。"
    assert data["confidence"] == 0.89
    assert data["session_id"] == "sess_123"
    assert len(data["sources"]) == 1
    assert len(data["retrieved_chunks"]) == 1
    assert data["sources"][0]["document_id"] == 10
    assert data["retrieved_chunks"][0]["chunk_type"] == "paragraph"