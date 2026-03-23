import argparse
import json
from typing import Any


from app.db import get_all_documents, init_db, reset_database
from app.services import (
    answer_question,
    get_chat_history,
    get_document_chunks,
    import_documents,
    import_single_document,
    index_document,
    summarize_document,
)


def print_json(data: Any):
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def format_section_path(section_path) -> str:
    if not section_path:
        return "-"
    if isinstance(section_path, str):
        return section_path
    if isinstance(section_path, list):
        return " > ".join(str(x) for x in section_path if str(x).strip())
    return str(section_path)


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


def cmd_init_db(_args):
    init_db()
    print("数据库初始化完成。")


def cmd_reset_db(args):
    if not args.yes:
        raise SystemExit("这是危险操作，请加 --yes 确认执行。")

    reset_database(keep_schema=not args.drop)

    if args.drop:
        print("数据库已删除并重建完成。")
    else:
        print("数据库数据已清空，表结构已保留。")


def cmd_import_folder(args):
    result = import_documents(args.folder)
    print_json(result)


def cmd_import_file(args):
    result = import_single_document(args.file_path)
    print_json(result)


def cmd_list_docs(_args):
    rows = get_all_documents()
    simplified = []
    for row in rows:
        simplified.append(
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "file_path": row.get("file_path"),
                "file_type": row.get("file_type"),
                "source_type": row.get("source_type"),
                "block_count": row.get("block_count"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            }
        )
    print_json(simplified)


def cmd_index(args):
    result = index_document(
        document_id=args.doc_id,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print_json(result)


def cmd_chunks(args):
    rows = get_document_chunks(args.doc_id)
    simplified = []

    for row in rows:
        metadata = row.get("metadata_json") or {}
        simplified.append(
            {
                "id": row.get("id"),
                "document_id": row.get("document_id"),
                "chunk_index": row.get("chunk_index"),
                "chunk_type": row.get("chunk_type"),
                "doc_title": row.get("doc_title"),
                "section_title": row.get("section_title"),
                "section_path": format_section_path(row.get("section_path")),
                "page_range": format_page_range(row.get("page_start"), row.get("page_end")),
                "token_count": row.get("token_count"),
                "chunk_hash": row.get("chunk_hash"),
                "preview": (row.get("search_text") or row.get("chunk_text") or "")[:260],
                "metadata": metadata,
            }
        )

    print_json(simplified)


def cmd_ask(args):
    result = answer_question(
        question=args.question,
        top_k=args.top_k,
        response_mode=args.response_mode,
        highlight=args.highlight,
        session_id=args.session_id,
        use_chat_context=args.use_chat_context,
    )
    print_json(result)


def cmd_summary(args):
    result = summarize_document(args.doc_id)
    print_json(result)


def cmd_chat_history(args):
    result = get_chat_history(session_id=args.session_id, limit=args.limit)
    print_json(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Knowledge Base CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init-db", help="初始化数据库")
    p_init.set_defaults(func=cmd_init_db)

    p_reset = subparsers.add_parser("reset-db", help="重置数据库")
    p_reset.add_argument(
        "--drop",
        action="store_true",
        help="删除所有表并重新初始化，而不是仅清空数据",
    )
    p_reset.add_argument(
        "--yes",
        action="store_true",
        help="确认执行危险操作",
    )
    p_reset.set_defaults(func=cmd_reset_db)

    p_import = subparsers.add_parser("import", help="导入文件夹中的文档")
    p_import.add_argument("folder", help="文件夹路径")
    p_import.set_defaults(func=cmd_import_folder)

    p_import_file = subparsers.add_parser("import-file", help="导入单个文档")
    p_import_file.add_argument("file_path", help="文件路径")
    p_import_file.set_defaults(func=cmd_import_file)

    p_list = subparsers.add_parser("list-docs", help="列出所有文档")
    p_list.set_defaults(func=cmd_list_docs)

    p_index = subparsers.add_parser("index", help="为指定文档建立索引")
    p_index.add_argument("--doc-id", type=int, required=True, help="文档 ID")
    p_index.add_argument(
        "--chunk-size",
        type=int,
        default=700,
        help="chunk 大小",
    )
    p_index.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="chunk overlap",
    )
    p_index.set_defaults(func=cmd_index)

    p_chunks = subparsers.add_parser("chunks", help="查看指定文档的 chunks")
    p_chunks.add_argument("--doc-id", type=int, required=True, help="文档 ID")
    p_chunks.set_defaults(func=cmd_chunks)

    p_ask = subparsers.add_parser("ask", help="基于知识库提问")
    p_ask.add_argument("question", help="问题")
    p_ask.add_argument("--top-k", type=int, default=5, help="召回 chunk 数量")
    p_ask.add_argument(
        "--response-mode",
        choices=["text", "structured"],
        default="text",
        help="回答模式",
    )
    p_ask.add_argument(
        "--highlight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否开启来源高亮",
    )
    p_ask.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="会话 ID，用于多轮上下文",
    )
    p_ask.add_argument(
        "--use-chat-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用聊天上下文改写 query",
    )
    p_ask.set_defaults(func=cmd_ask)

    p_summary = subparsers.add_parser("summary", help="摘要指定文档")
    p_summary.add_argument("--doc-id", type=int, required=True, help="文档 ID")
    p_summary.set_defaults(func=cmd_summary)

    p_history = subparsers.add_parser("chat-history", help="查看会话历史")
    p_history.add_argument("--session-id", type=str, required=True, help="会话 ID")
    p_history.add_argument("--limit", type=int, default=20, help="返回消息条数")
    p_history.set_defaults(func=cmd_chat_history)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()