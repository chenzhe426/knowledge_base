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


def cmd_init_db():
    init_db()
    print("数据库初始化完成")


def cmd_import(folder: str):
    result = import_documents(folder)
    print_json(result)


def cmd_import_file(file_path: str):
    result = import_single_document(file_path)
    print_json(result)


def cmd_list_docs():
    print_json(list_documents())


def cmd_index(doc_id: int, chunk_size: int, overlap: int):
    result = index_document(doc_id, chunk_size=chunk_size, overlap=overlap)
    print_json(result)


def cmd_chunks(doc_id: int):
    result = get_document_chunks(doc_id)
    print_json(result)


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

    subparsers.add_parser("list-docs", help="查看文档列表")

    p_index = subparsers.add_parser("index", help="为某篇文档切 chunk 并生成 embedding")
    p_index.add_argument("doc_id", type=int)
    p_index.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p_index.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)

    p_chunks = subparsers.add_parser("chunks", help="查看某篇文档的 chunks")
    p_chunks.add_argument("doc_id", type=int)

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
        cmd_list_docs()
    elif args.command == "index":
        cmd_index(args.doc_id, args.chunk_size, args.overlap)
    elif args.command == "chunks":
        cmd_chunks(args.doc_id)
    elif args.command == "ask":
        cmd_ask(args.question, args.top_k)
    elif args.command == "summary":
        cmd_summary(args.text)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()