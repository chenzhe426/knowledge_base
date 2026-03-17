import argparse
import logging
from pprint import pprint

from app.config import (
    DATA_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
)
from app.db import (
    init_db,
    get_all_documents,
    search_documents,
    get_document_by_id,
)
from app.services import (
    import_documents,
    summarize_text,
    index_document,
    get_document_chunks,
    retrieve_chunks,
    answer_question,
)
from app.utils import setup_logger


setup_logger()


def show_document(doc_id: int):
    row = get_document_by_id(doc_id)
    if not row:
        print("文档不存在")
        return

    print("=" * 80)
    print(f"ID        : {row['id']}")
    print(f"标题      : {row['title']}")
    print(f"文件路径  : {row['file_path']}")
    print(f"创建时间  : {row['created_at']}")
    print("-" * 80)
    print(row["content"])
    print("=" * 80)


def list_documents():
    docs = get_all_documents(limit=100, offset=0)
    if not docs:
        print("当前没有文档")
        return

    for doc in docs:
        print(
            f"[{doc['id']}] {doc['title']} | "
            f"{doc['file_path']} | {doc['created_at']}"
        )


def search_docs(keyword: str):
    docs = search_documents(keyword, limit=20, offset=0)
    if not docs:
        print("未找到匹配文档")
        return

    for doc in docs:
        print(
            f"[{doc['id']}] {doc['title']} | "
            f"{doc['file_path']} | {doc['created_at']}"
        )


def summarize_document(doc_id: int):
    row = get_document_by_id(doc_id)
    if not row:
        print("文档不存在")
        return

    summary = summarize_text(row["content"])
    print("=" * 80)
    print(f"文档: {row['title']}")
    print("-" * 80)
    print(summary)
    print("=" * 80)


def build_index(doc_id: int, chunk_size: int, overlap: int):
    result = index_document(
        doc_id=doc_id,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    print("=" * 80)
    print(f"文档索引完成: {result['title']}")
    print(f"document_id: {result['document_id']}")
    print(f"chunk_count : {result['chunk_count']}")
    print("-" * 80)

    for chunk in result["chunks"][:10]:
        print(
            f"chunk={chunk['chunk_index']}, "
            f"range=({chunk['char_start']}, {chunk['char_end']}), "
            f"preview={chunk['text_preview']}"
        )

    if result["chunk_count"] > 10:
        print("... 仅展示前 10 个 chunk")
    print("=" * 80)


def show_chunks(doc_id: int):
    chunks = get_document_chunks(doc_id)
    if not chunks:
        print("该文档暂无 chunk，请先执行 index")
        return

    print(f"文档 {doc_id} 共 {len(chunks)} 个 chunks")
    print("=" * 80)
    for chunk in chunks:
        print(
            f"[chunk {chunk['chunk_index']}] "
            f"range=({chunk['char_start']}, {chunk['char_end']})"
        )
        print(chunk["text"])
        print("-" * 80)


def retrieve(query: str, top_k: int):
    results = retrieve_chunks(query=query, top_k=top_k)
    if not results:
        print("没有检索到结果")
        return

    print("=" * 80)
    print(f"Query: {query}")
    print(f"Top K: {top_k}")
    print("-" * 80)

    for i, item in enumerate(results, start=1):
        print(
            f"[{i}] score={item['score']} | "
            f"doc={item['document_id']} | "
            f"title={item['title']} | "
            f"chunk={item['chunk_index']}"
        )
        print(item["text"])
        print("-" * 80)

    print("=" * 80)


def ask(query: str, top_k: int):
    result = answer_question(query=query, top_k=top_k)

    print("=" * 80)
    print("问题：")
    print(result["question"])
    print("-" * 80)
    print("回答：")
    print(result["answer"])
    print("-" * 80)
    print("引用来源：")

    for i, source in enumerate(result["sources"], start=1):
        print(
            f"[来源{i}] score={source['score']} | "
            f"doc={source['document_id']} | "
            f"title={source['title']} | "
            f"chunk={source['chunk_index']}"
        )
        print(source["text"])
        print("-" * 80)

    print("=" * 80)


def main():
    logging.info("程序启动")
    init_db()

    parser = argparse.ArgumentParser(description="Knowledge Base RAG CLI")
    subparsers = parser.add_subparsers(dest="command")

    import_parser = subparsers.add_parser("import", help="导入文件夹中的文档")
    import_parser.add_argument("folder", nargs="?", default=DATA_DIR)

    subparsers.add_parser("list", help="列出所有文档")

    search_parser = subparsers.add_parser("search", help="关键词搜索文档")
    search_parser.add_argument("keyword")

    show_parser = subparsers.add_parser("show", help="显示单个文档内容")
    show_parser.add_argument("id", type=int)

    summary_parser = subparsers.add_parser("summary", help="总结单个文档")
    summary_parser.add_argument("id", type=int)

    index_parser = subparsers.add_parser("index", help="为文档构建 RAG 索引")
    index_parser.add_argument("id", type=int)
    index_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )
    index_parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
    )

    chunks_parser = subparsers.add_parser("chunks", help="查看文档 chunks")
    chunks_parser.add_argument("id", type=int)

    retrieve_parser = subparsers.add_parser("retrieve", help="只做召回调试")
    retrieve_parser.add_argument("query")
    retrieve_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    ask_parser = subparsers.add_parser("ask", help="RAG 问答")
    ask_parser.add_argument("query")
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    args = parser.parse_args()

    try:
        if args.command == "import":
            doc_ids = import_documents(args.folder)
            print(f"导入完成，共导入 {len(doc_ids)} 个文档")
            print("document_ids =", doc_ids)

        elif args.command == "list":
            list_documents()

        elif args.command == "search":
            search_docs(args.keyword)

        elif args.command == "show":
            show_document(args.id)

        elif args.command == "summary":
            summarize_document(args.id)

        elif args.command == "index":
            if args.overlap >= args.chunk_size:
                print("错误：overlap 必须小于 chunk-size")
                return
            build_index(
                doc_id=args.id,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )

        elif args.command == "chunks":
            show_chunks(args.id)

        elif args.command == "retrieve":
            retrieve(args.query, args.top_k)

        elif args.command == "ask":
            ask(args.query, args.top_k)

        else:
            parser.print_help()

    except Exception as e:
        logging.exception("执行失败")
        print(f"执行失败: {e}")


if __name__ == "__main__":
    main()