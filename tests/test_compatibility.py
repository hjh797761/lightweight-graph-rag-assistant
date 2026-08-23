import importlib


class FakeChunk:
    id = "doc::chunk_000000"


class FakeResult:
    path = "full召回-top-1 -> 选出-top-1"
    chunks = [FakeChunk()]
    context = "[S1] 证据"


class FakeService:
    def load(self):
        return {"doc": 1}

    def retrieve(self, query, doc_scope=None, profile="full"):
        return FakeResult()

    def generate_answer(self, query, context):
        return "回答"

    def add_document(self, file_path, batch_size=20):
        return 1, 1


def test_legacy_public_functions_exist():
    legacy = importlib.import_module("graphrag_assistant")
    for name in (
        "load_knowledge_base",
        "retrieve",
        "retrieve_semantic",
        "generate_answer",
        "add_document_to_tree",
    ):
        assert callable(getattr(legacy, name))


def test_legacy_retrieve_returns_three_values(monkeypatch):
    legacy = importlib.import_module("graphrag_assistant")
    monkeypatch.setattr(legacy, "_service", FakeService())

    path, selected, context = legacy.retrieve("问题")

    assert path == "full召回-top-1 -> 选出-top-1"
    assert selected == "doc::chunk_000000"
    assert "[S1]" in context


def test_legacy_answer_and_ingestion_delegate(monkeypatch):
    legacy = importlib.import_module("graphrag_assistant")
    monkeypatch.setattr(legacy, "_service", FakeService())
    assert legacy.generate_answer("问题", "[S1] 证据") == "回答"
    assert legacy.add_document_to_tree("doc.txt", batch_size=3) == (1, 1)
