"""Backward-compatible entry point and interactive CLI."""

from graphrag.service import LazyService, build_default_service


_service = LazyService(build_default_service)


def load_knowledge_base():
    progress = _service.load()
    print(f"✅ 知识库加载成功，共 {len(_service.list_documents())} 个文档")
    return progress


def retrieve(query, doc_scope=None):
    result = _service.retrieve(query, doc_scope=doc_scope, profile="full")
    return result.path, ",".join(chunk.id for chunk in result.chunks), result.context


def retrieve_semantic(query, doc_scope=None, mode="embedding_graph"):
    profile = {
        "embedding": "vector",
        "vector": "vector",
        "embedding_graph": "full",
        "vector_graph": "vector_graph",
        "full": "full",
    }.get(mode, "full")
    result = _service.retrieve(query, doc_scope=doc_scope, profile=profile)
    return result.path, ",".join(chunk.id for chunk in result.chunks), result.context


def generate_answer(query, context):
    return _service.generate_answer(query, context)


def add_document_to_tree(file_path, batch_size=20):
    processed, total = _service.add_document(file_path, batch_size=batch_size)
    if processed < total:
        print(f"⏸ 本批处理完成，进度 {processed}/{total}；再次上传同一文件可继续")
    else:
        print(f"✅ 文件处理完成，共 {total} 个片段")
    return processed, total


def reset_knowledge_base(make_backup=True):
    backup = _service.reset(make_backup=make_backup)
    if backup:
        print(f"✅ 已备份旧知识库 → {backup}")
    print("✅ 知识库已清空，可以重新导入文档")


def _print_tree():
    tree = _service.list_tree()
    if not tree:
        print("（暂无知识结构）")
    for doc_id, chapters in tree.items():
        print(f"\n文档：{doc_id}")
        for chapter, concepts in chapters.items():
            print(f"  {chapter}:")
            for concept in concepts:
                print(f"    - {concept}")


def _print_documents():
    documents = _service.list_documents()
    if not documents:
        print("（暂无文档索引）")
    for document in documents:
        print(f"- {document['id']}")
        print(f"  名称：{document['name']}")
        print(f"  路径：{document['path']}")
        print(f"  已处理片段：{document['processed_chunks']}")


def main():
    load_knowledge_base()
    while True:
        print("\n请选择操作:")
        print("1. 上传资料")
        print("2. 提问")
        print("3. 查看知识树")
        print("4. 查看文档列表")
        print("9. 备份并清空知识库")
        print("0. 退出")
        choice = input("请输入选项：").strip()
        if choice == "1":
            add_document_to_tree(input("请输入文件路径：").strip())
        elif choice == "2":
            query = input("请输入你的问题：").strip()
            path, selected, context = retrieve(query)
            print(f"\n【检索路径】\n{path}")
            if selected:
                print(f"【命中片段】\n{selected}")
            print(f"\n【回答】\n{generate_answer(query, context)}")
        elif choice == "3":
            print("\n当前知识树")
            _print_tree()
        elif choice == "4":
            print("\n当前文档列表:")
            _print_documents()
        elif choice == "9":
            if input("这会备份后清空知识库。确认请输入 RESET：").strip() == "RESET":
                reset_knowledge_base(make_backup=True)
            else:
                print("已取消清空操作")
        elif choice == "0":
            break
        else:
            print("输入错误，请重试")


if __name__ == "__main__":
    main()
