import argparse
import json

from app.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K
from app.db import (
    get_all_documents,
    get_document_by_id,
    init_db,
    search_documents,
)
from app.services import (
    answer_question,
    get_document_chunks,
    import_documents,
    import_single_document,
    index_document,
    retrieve_chunks,
    summarize_text,
)


def print_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def format_section_path(section_path) -> str:
    if not section_path:
        return "-"
    if isinstance(section_path, str):
        return section_path
    return " > ".join(section_path)


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


def cmd_import(folder: str):
    result = import_documents(folder)

    print(f"总计解析文档: {result['total']}")
    print(f"成功导入: {result['imported_count']}")
    print(f"失败数量: {result['failed_count']}")
    print()

    if result["imported"]:
        print("导入成功的文档：")
        for item in result["imported"]:
            print(
                f"- [ID={item['id']}] {item['title']} | "
                f"{item.get('file_type') or '-'} | {item.get('file_path') or '-'} | "
                f"字符数={item['char_count']} | block数={item['block_count']}"
            )
        print()

    if result["failed"]:
        print("导入失败的文档：")
        for item in result["failed"]:
            print(f"- {item.get('file_path')} | 原因: {item['reason']}")


def cmd_import_file(file_path: str):
    result = import_single_document(file_path=file_path)

    print("导入成功：")
    print(f"ID: {result['id']}")
    print(f"标题: {result['title']}")
    print(f"路径: {result.get('file_path') or '-'}")
    print(f"文件类型: {result.get('file_type') or '-'}")
    print(f"来源类型: {result.get('source_type') or '-'}")
    print(f"字符数: {result['char_count']}")
    print(f"Block 数: {result['block_count']}")


def cmd_list():
    docs = get_all_documents()
    if not docs:
        print("暂无文档。")
        return

    for row in docs:
        print(
            f"[{row['id']}] {row['title']} | "
            f"path={row.get('file_path') or '-'} | "
            f"type={row.get('file_type') or '-'} | "
            f"source={row.get('source_type') or '-'} | "
            f"blocks={row.get('block_count', 0)}"
        )


def cmd_search(query: str):
    docs = search_documents(query)
    if not docs:
        print("没有搜索到相关文档。")
        return

    for row in docs:
        print(
            f"[{row['id']}] {row['title']} | "
            f"path={row.get('file_path') or '-'} | "
            f"type={row.get('file_type') or '-'} | "
            f"source={row.get('source_type') or '-'} | "
            f"blocks={row.get('block_count', 0)}"
        )


def cmd_show(doc_id: int, show_blocks: bool = False):
    row = get_document_by_id(doc_id)
    if not row:
        print("文档不存在。")
        return

    print(f"ID: {row['id']}")
    print(f"标题: {row['title']}")
    print(f"路径: {row.get('file_path') or '-'}")
    print(f"文件类型: {row.get('file_type') or '-'}")
    print(f"来源类型: {row.get('source_type') or '-'}")
    print(f"Block 数: {row.get('block_count', 0)}")
    print("-" * 80)
    print(row.get("content", ""))

    if show_blocks:
        print()
        print("=" * 80)
        print("Blocks JSON：")
        print(row.get("blocks_json") or "[]")


def cmd_summary(doc_id: int):
    row = get_document_by_id(doc_id)
    if not row:
        print("文档不存在。")
        return

    summary = summarize_text(row["content"])
    print("文档总结：")
    print(summary)


def cmd_index(doc_id: int, chunk_size: int, overlap: int):
    result = index_document(doc_id=doc_id, chunk_size=chunk_size, overlap=overlap)

    print(f"已完成索引: 文档ID={result['document_id']} | 标题={result['title']}")
    print(f"共生成 {result['chunk_count']} 个 chunks")
    print()

    for item in result["chunks"]:
        section_path = format_section_path(item.get("section_path"))
        page_range = format_page_range(item.get("page_start"), item.get("page_end"))

        print(
            f"- chunk_index={item['chunk_index']} | "
            f"type={item.get('chunk_type') or '-'} | "
            f"section={section_path} | "
            f"pages={page_range} | "
            f"blocks=({item.get('block_start_order')}, {item.get('block_end_order')}) | "
            f"tokens={item.get('token_count', 0)}"
        )
        print(f"  preview: {item.get('text_preview', '')}")
        print()


def cmd_chunks(doc_id: int):
    chunks = get_document_chunks(doc_id)
    if not chunks:
        print("该文档暂无 chunks。")
        return

    for item in chunks:
        section_path = format_section_path(item.get("section_path"))
        page_range = format_page_range(item.get("page_start"), item.get("page_end"))

        print("=" * 100)
        print(
            f"chunk_id={item['chunk_id']} | "
            f"chunk_index={item['chunk_index']} | "
            f"type={item.get('chunk_type') or '-'} | "
            f"section={section_path} | "
            f"pages={page_range} | "
            f"blocks=({item.get('block_start_order')}, {item.get('block_end_order')}) | "
            f"tokens={item.get('token_count', 0)} | "
            f"char_range=({item.get('char_start')}, {item.get('char_end')})"
        )
        print(item["text"])


def cmd_retrieve(query: str, top_k: int):
    results = retrieve_chunks(query=query, top_k=top_k)
    if not results:
        print("没有检索到相关片段。")
        return

    for item in results:
        section_path = format_section_path(item.get("section_path"))
        page_range = format_page_range(item.get("page_start"), item.get("page_end"))

        print("=" * 100)
        print(
            f"score={item['score']} | "
            f"doc_id={item['document_id']} | "
            f"title={item['title']} | "
            f"chunk_index={item['chunk_index']} | "
            f"type={item.get('chunk_type') or '-'} | "
            f"section={section_path} | "
            f"pages={page_range} | "
            f"tokens={item.get('token_count', 0)}"
        )
        print(item["text"])


def cmd_ask(query: str, top_k: int):
    result = answer_question(query=query, top_k=top_k)

    print("问题：")
    print(result["question"])
    print()

    print("回答：")
    print(result["answer"])
    print()

    if result["sources"]:
        print("来源片段：")
        for item in result["sources"]:
            section_path = format_section_path(item.get("section_path"))
            page_range = format_page_range(item.get("page_start"), item.get("page_end"))

            print("=" * 100)
            print(
                f"score={item['score']} | "
                f"doc_id={item['document_id']} | "
                f"title={item['title']} | "
                f"chunk_index={item['chunk_index']} | "
                f"type={item.get('chunk_type') or '-'} | "
                f"section={section_path} | "
                f"pages={page_range} | "
                f"tokens={item.get('token_count', 0)}"
            )
            print(item["text"])


def build_parser():
    parser = argparse.ArgumentParser(description="Knowledge Base CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init-db", help="初始化或升级数据库表结构")

    parser_import = subparsers.add_parser("import", help="从文件夹批量导入文档")
    parser_import.add_argument("folder", type=str, help="文档文件夹路径")

    parser_import_file = subparsers.add_parser("import-file", help="导入单个文档")
    parser_import_file.add_argument("file_path", type=str, help="文件路径")

    subparsers.add_parser("list", help="列出所有文档")

    parser_search = subparsers.add_parser("search", help="搜索文档")
    parser_search.add_argument("query", type=str, help="搜索关键词")

    parser_show = subparsers.add_parser("show", help="查看文档内容")
    parser_show.add_argument("doc_id", type=int, help="文档 ID")
    parser_show.add_argument(
        "--show-blocks",
        action="store_true",
        help="同时打印 documents.blocks_json",
    )

    parser_summary = subparsers.add_parser("summary", help="总结文档")
    parser_summary.add_argument("doc_id", type=int, help="文档 ID")

    parser_index = subparsers.add_parser("index", help="为文档建立向量索引")
    parser_index.add_argument("doc_id", type=int, help="文档 ID")
    parser_index.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="chunk 大小（当前更接近 max_tokens 的近似控制）",
    )
    parser_index.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="保留参数，当前主要作为语义上下文策略配置",
    )

    parser_chunks = subparsers.add_parser("chunks", help="查看文档 chunks")
    parser_chunks.add_argument("doc_id", type=int, help="文档 ID")

    parser_retrieve = subparsers.add_parser("retrieve", help="检索相关片段")
    parser_retrieve.add_argument("query", type=str, help="检索问题")
    parser_retrieve.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="返回片段数量",
    )

    parser_ask = subparsers.add_parser("ask", help="基于知识库问答")
    parser_ask.add_argument("query", type=str, help="问题")
    parser_ask.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="召回片段数量",
    )

    parser_debug_retrieve = subparsers.add_parser(
        "retrieve-json",
        help="以 JSON 形式输出检索结果，便于调试",
    )
    parser_debug_retrieve.add_argument("query", type=str, help="检索问题")
    parser_debug_retrieve.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="返回片段数量",
    )

    parser_debug_chunks = subparsers.add_parser(
        "chunks-json",
        help="以 JSON 形式输出某文档 chunks，便于调试",
    )
    parser_debug_chunks.add_argument("doc_id", type=int, help="文档 ID")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "init-db":
        init_db()
        print("数据库初始化/升级完成。")
        return

    init_db()

    if args.command == "import":
        cmd_import(args.folder)
    elif args.command == "import-file":
        cmd_import_file(args.file_path)
    elif args.command == "list":
        cmd_list()
    elif args.command == "search":
        cmd_search(args.query)
    elif args.command == "show":
        cmd_show(args.doc_id, args.show_blocks)
    elif args.command == "summary":
        cmd_summary(args.doc_id)
    elif args.command == "index":
        cmd_index(args.doc_id, args.chunk_size, args.overlap)
    elif args.command == "chunks":
        cmd_chunks(args.doc_id)
    elif args.command == "retrieve":
        cmd_retrieve(args.query, args.top_k)
    elif args.command == "ask":
        cmd_ask(args.query, args.top_k)
    elif args.command == "retrieve-json":
        print_json(retrieve_chunks(query=args.query, top_k=args.top_k))
    elif args.command == "chunks-json":
        print_json(get_document_chunks(args.doc_id))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()