import argparse
import json

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
                f"{item['file_type']} | {item['file_path']} | "
                f"字符数={item['char_count']} | block数={item['block_count']}"
            )
        print()

    if result["failed"]:
        print("导入失败的文档：")
        for item in result["failed"]:
            print(f"- {item.get('file_path')} | 原因: {item['reason']}")


def cmd_import_file(file_path: str, source_type: str):
    result = import_single_document(file_path=file_path, source_type=source_type)

    print("导入成功：")
    print(f"ID: {result['id']}")
    print(f"标题: {result['title']}")
    print(f"路径: {result['file_path']}")
    print(f"文件类型: {result['file_type']}")
    print(f"来源类型: {result['source_type']}")
    print(f"字符数: {result['char_count']}")
    print(f"Block 数: {result['block_count']}")


def cmd_list():
    docs = get_all_documents()
    if not docs:
        print("暂无文档。")
        return

    for row in docs:
        print(f"[{row['id']}] {row['title']} ({row.get('file_path')})")


def cmd_search(query: str):
    docs = search_documents(query)
    if not docs:
        print("没有搜索到相关文档。")
        return

    for row in docs:
        print(f"[{row['id']}] {row['title']} ({row.get('file_path')})")


def cmd_show(doc_id: int):
    row = get_document_by_id(doc_id)
    if not row:
        print("文档不存在。")
        return

    print(f"ID: {row['id']}")
    print(f"标题: {row['title']}")
    print(f"路径: {row.get('file_path')}")
    print("-" * 80)
    print(row.get("content", ""))


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
        print(
            f"- chunk_index={item['chunk_index']} "
            f"[{item['char_start']}, {item['char_end']}] "
            f"{item['text_preview']}"
        )


def cmd_chunks(doc_id: int):
    chunks = get_document_chunks(doc_id)
    if not chunks:
        print("该文档暂无 chunks。")
        return

    for item in chunks:
        print("=" * 80)
        print(
            f"chunk_id={item['chunk_id']} | "
            f"chunk_index={item['chunk_index']} | "
            f"char_range=({item['char_start']}, {item['char_end']})"
        )
        print(item["text"])


def cmd_retrieve(query: str, top_k: int):
    results = retrieve_chunks(query=query, top_k=top_k)
    if not results:
        print("没有检索到相关片段。")
        return

    for item in results:
        print("=" * 80)
        print(
            f"score={item['score']} | "
            f"doc_id={item['document_id']} | "
            f"title={item['title']} | "
            f"chunk_index={item['chunk_index']}"
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
            print("=" * 80)
            print(
                f"score={item['score']} | "
                f"doc_id={item['document_id']} | "
                f"title={item['title']} | "
                f"chunk_index={item['chunk_index']}"
            )
            print(item["text"])


def build_parser():
    parser = argparse.ArgumentParser(description="Knowledge Base CLI")
    subparsers = parser.add_subparsers(dest="command")

    # import folder
    parser_import = subparsers.add_parser("import", help="从文件夹批量导入文档")
    parser_import.add_argument("folder", type=str, help="文档文件夹路径")

    # import single file
    parser_import_file = subparsers.add_parser("import-file", help="导入单个文档")
    parser_import_file.add_argument("file_path", type=str, help="文件路径")
    parser_import_file.add_argument(
        "--source-type",
        type=str,
        default="upload",
        help="来源类型，例如 upload/folder/url",
    )

    # list
    subparsers.add_parser("list", help="列出所有文档")

    # search
    parser_search = subparsers.add_parser("search", help="搜索文档")
    parser_search.add_argument("query", type=str, help="搜索关键词")

    # show
    parser_show = subparsers.add_parser("show", help="查看文档内容")
    parser_show.add_argument("doc_id", type=int, help="文档 ID")

    # summary
    parser_summary = subparsers.add_parser("summary", help="总结文档")
    parser_summary.add_argument("doc_id", type=int, help="文档 ID")

    # index
    parser_index = subparsers.add_parser("index", help="为文档建立向量索引")
    parser_index.add_argument("doc_id", type=int, help="文档 ID")
    parser_index.add_argument("--chunk-size", type=int, default=500, help="chunk 大小")
    parser_index.add_argument("--overlap", type=int, default=100, help="chunk 重叠大小")

    # chunks
    parser_chunks = subparsers.add_parser("chunks", help="查看文档 chunks")
    parser_chunks.add_argument("doc_id", type=int, help="文档 ID")

    # retrieve
    parser_retrieve = subparsers.add_parser("retrieve", help="检索相关片段")
    parser_retrieve.add_argument("query", type=str, help="检索问题")
    parser_retrieve.add_argument("--top-k", type=int, default=3, help="返回片段数量")

    # ask
    parser_ask = subparsers.add_parser("ask", help="基于知识库问答")
    parser_ask.add_argument("query", type=str, help="问题")
    parser_ask.add_argument("--top-k", type=int, default=3, help="召回片段数量")

    return parser


def main():
    init_db()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "import":
        cmd_import(args.folder)

    elif args.command == "import-file":
        cmd_import_file(args.file_path, args.source_type)

    elif args.command == "list":
        cmd_list()

    elif args.command == "search":
        cmd_search(args.query)

    elif args.command == "show":
        cmd_show(args.doc_id)

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

    else:
        parser.print_help()


if __name__ == "__main__":
    main()