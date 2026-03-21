from __future__ import annotations

import argparse
import json

from app.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K
from app.db import init_db
from app.services import (
    answer_question,
    get_document_chunks,
    import_documents,
    import_single_document,
    index_document,
    list_documents,
    summarize_text,
)


def print_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def format_section_path(section_path) -> str:
    if not section_path:
        return "-"
    if isinstance(section_path, str):
        return section_path
    return " > ".join(str(x) for x in section_path if str(x).strip()) or "-"


def format_page_range(page_start, page_end) -> str:
    if page_start is None and page_end is None:
        return "-"
    if page_start is not None and page_end is not None:
        if page_start == page_end:
            return str(page_start)
        return f"{page_start}-{page_end}"
    if page_start is not None:
        return str(page_start)
    return str(page_end)


def truncate_text(text: str, max_len: int = 160) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def cmd_init_db():
    init_db()
    print("数据库初始化完成")


def cmd_import(folder: str):
    result = import_documents(folder)
    print_json(result)


def cmd_import_file(file_path: str):
    result = import_single_document(file_path)
    print_json(result)


def cmd_list_docs(as_json: bool = False):
    docs = list_documents()
    if as_json:
        print_json(docs)
        return

    if not docs:
        print("暂无文档")
        return

    print(f"文档总数: {len(docs)}")
    print("-" * 120)
    print(f"{'ID':<6}{'标题':<28}{'类型':<10}{'字符数':<10}{'Block数':<10}{'创建时间'}")
    print("-" * 120)

    for doc in docs:
        title = truncate_text(doc.get("title", ""), 26)
        file_type = doc.get("file_type") or "-"
        char_count = doc.get("char_count") or 0
        block_count = doc.get("block_count") or 0
        created_at = doc.get("created_at") or "-"
        print(f"{doc['id']:<6}{title:<28}{file_type:<10}{char_count:<10}{block_count:<10}{created_at}")

    print("-" * 120)


def cmd_index(doc_id: int, chunk_size: int, overlap: int):
    result = index_document(doc_id, chunk_size=chunk_size, overlap=overlap)
    print_json(result)


def cmd_chunks(
    doc_id: int,
    full: bool = False,
    as_json: bool = False,
    limit: int | None = None,
):
    result = get_document_chunks(doc_id)

    if as_json:
        print_json(result)
        return

    if not result:
        print(f"文档 {doc_id} 暂无 chunks")
        return

    rows = result[:limit] if limit and limit > 0 else result

    print(f"文档ID: {doc_id}")
    print(f"Chunk总数: {len(result)}")
    if limit and limit > 0:
        print(f"当前展示: {len(rows)}")
    print("=" * 120)

    for idx, chunk in enumerate(rows, start=1):
        chunk_id = chunk.get("id")
        chunk_type = chunk.get("chunk_type") or "-"
        section_title = chunk.get("section_title") or "-"
        section_path = format_section_path(chunk.get("section_path"))
        page_range = format_page_range(chunk.get("page_start"), chunk.get("page_end"))
        token_count = chunk.get("token_count") or "-"
        preview = chunk.get("preview") or ""

        print(f"[{idx}] Chunk ID: {chunk_id}")
        print(f"类型       : {chunk_type}")
        print(f"章节标题   : {section_title}")
        print(f"章节路径   : {section_path}")
        print(f"页码范围   : {page_range}")
        print(f"Token估算  : {token_count}")
        print("内容       :")
        if full:
            print(preview)
        else:
            print(truncate_text(preview, 220))
        print("-" * 120)

    if limit and len(result) > len(rows):
        print(f"还有 {len(result) - len(rows)} 个 chunk 未展示，可去掉 --limit 查看全部。")


def cmd_ask(question: str, top_k: int):
    result = answer_question(question, top_k=top_k)
    print_json(result)


def cmd_summary(text: str):
    result = summarize_text(text)
    print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Knowledge Base CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="初始化数据库")

    p_import = subparsers.add_parser("import", help="导入文件夹中的文档")
    p_import.add_argument("folder", type=str)

    p_import_file = subparsers.add_parser("import-file", help="导入单个文档")
    p_import_file.add_argument("file_path", type=str)

    p_list = subparsers.add_parser("list-docs", help="查看文档列表")
    p_list.add_argument("--json", action="store_true", help="按 JSON 输出")

    p_index = subparsers.add_parser("index", help="为某篇文档切 chunk 并生成 embedding")
    p_index.add_argument("doc_id", type=int)
    p_index.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p_index.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)

    p_chunks = subparsers.add_parser("chunks", help="查看某篇文档的 chunks")
    p_chunks.add_argument("doc_id", type=int)
    p_chunks.add_argument("--full", action="store_true", help="显示完整内容")
    p_chunks.add_argument("--json", action="store_true", help="按 JSON 输出")
    p_chunks.add_argument("--limit", type=int, default=None, help="限制展示数量")

    p_ask = subparsers.add_parser("ask", help="提问")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    p_summary = subparsers.add_parser("summary", help="总结文本")
    p_summary.add_argument("text", type=str)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init-db":
        cmd_init_db()
    elif args.command == "import":
        cmd_import(args.folder)
    elif args.command == "import-file":
        cmd_import_file(args.file_path)
    elif args.command == "list-docs":
        cmd_list_docs(as_json=args.json)
    elif args.command == "index":
        cmd_index(args.doc_id, args.chunk_size, args.overlap)
    elif args.command == "chunks":
        cmd_chunks(
            args.doc_id,
            full=args.full,
            as_json=args.json,
            limit=args.limit,
        )
    elif args.command == "ask":
        cmd_ask(args.question, args.top_k)
    elif args.command == "summary":
        cmd_summary(args.text)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()