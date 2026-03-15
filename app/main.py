import argparse
from app.db import init_db, get_all_documents, search_documents, get_connection
from app.services import import_documents


def show_document(doc_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content, file_path, created_at FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        print(row)
    else:
        print("文档不存在")


def main():
    init_db()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("folder")

    subparsers.add_parser("list")

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("keyword")

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("id", type=int)

    args = parser.parse_args()

    if args.command == "import":
        import_documents(args.folder)
        print("导入完成")
    elif args.command == "list":
        docs = get_all_documents()
        for doc in docs:
            print(doc)
    elif args.command == "search":
        docs = search_documents(args.keyword)
        for doc in docs:
            print(doc)
    elif args.command == "show":
        show_document(args.id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()