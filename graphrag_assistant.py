"""Backward-compatible entry point and interactive CLI."""

from graphrag.service import LazyService, build_default_service


_service = LazyService(build_default_service)


def load_knowledge_base():
    progress = _service.load()
    print(f"✅ 知识库加载成功，共 {len(_service.list_documents())} 个文档")
    return progress


def retrieve(query, doc_scope=None):
    result = _service.retrieve(query, doc_scope=doc_scope)
    return result.path, ",".join(chunk.id for chunk in result.chunks), result.context


def retrieve_semantic(query, doc_scope=None, mode="embedding_graph"):
    profile = {
        "embedding": "vector",
        "vector": "vector",
        "embedding_graph": "full",
        "vector_graph": "vector_graph",
        "full": "full",
        "vector_keyword": "vector_keyword",
        "vector_topic_graph": "vector_topic_graph",
        "evidence": "evidence",
    }.get(mode)
    if profile is None:
        raise ValueError(f"未知检索模式: {mode}")
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
    print("轻量证据关联助手")
    load_knowledge_base()
    while True:
        print("\n请选择操作:")
        print("1. 上传资料")
        print("2. 提问")
        print("3. 查看文档结构")
        print("4. 查看文档列表")
        print("9. 备份并清空知识库")
        print("0. 退出")
        choice = input("请输入选项：").strip()
        if choice == "1":
            add_document_to_tree(input("请输入文件路径：").strip())
        elif choice == "2":
            query = input("请输入你的问题：").strip()
            result = _service.retrieve(query)
            _print_retrieval(result)
            print(f"\n【回答】\n{generate_answer(query, result.context)}")
        elif choice == "3":
            print("\n当前文档结构")
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


def _print_retrieval(result):
    print(f"\n【检索路径】\n{result.path}")
    if result.chunks:
        print("【命中片段】\n" + ",".join(chunk.id for chunk in result.chunks))
    selection = result.selection
    if selection is not None:
        states = {"ok": "已选出证据", "no_candidates": "无候选证据", "budget_exhausted": "上下文预算不足"}
        print(f"【证据状态】{states.get(selection.status, selection.status)}")
        print(f"【上下文预算已用】{selection.budget_used} {selection.budget_unit}")
        for chunk, role in zip(selection.chunks, selection.roles):
            label = {"core": "核心证据", "supplement": "补充证据"}.get(role, role)
            partial = "；原文片段不完整" if chunk.partial else ""
            print(f"  {label}: {chunk.id}；来源：{chunk.locator}{partial}")
            for reason in selection.reasons.get(chunk.id, []):
                kind = {"explicit_reference": "显式引用", "adjacent": "同节相邻"}.get(reason.get("kind"), reason.get("kind", ""))
                print(f"    原因：{kind} {reason.get('basis', '')}；来自 {reason.get('source_id', '')}")
        limitations = {"no_links": "无可用关联", "section_membership_only": "仅有章节归属，无可用关联补充",
            "unresolved_only": "引用尚未解析，无可用关联补充", "no_usable_links": "无可用关联",
            "partial_coverage": "引用目标仅部分覆盖", "missing": "引用目标缺失", "ambiguous": "引用目标不唯一",
            "budget_skipped": "因预算不足未纳入", "cap_skipped": "因片段数量上限未纳入",
            "disabled_supplements": "补充证据已禁用", "invalid_target_scope": "关联目标不在允许的来源范围"}
        for decision in selection.decisions:
            status = decision.get("status")
            if status in limitations:
                print(f"  关联／限制：{limitations[status]} {decision.get('chunk_id', decision.get('source_id', ''))}")


if __name__ == "__main__":
    main()
