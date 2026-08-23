from pathlib import Path

from scripts.retrieval_eval import evaluate, load_fixture, write_markdown


class FakeChunk:
    def __init__(self, text):
        self.id = "doc::chunk_000000"
        self.clean_text = text


class FakeResult:
    def __init__(self, text, scope, top_k):
        self.path = "test"
        self.chunks = [FakeChunk(text)]
        self.context = f"[S1] {text}"
        self.requested_scope = scope
        self.requested_top_k = top_k


class FakeService:
    def retrieve(self, query, doc_scope=None, profile="full", top_k=None):
        return FakeResult("温度传感器、控制器和冷却阀", doc_scope, top_k)


def test_public_fixture_has_no_absolute_paths():
    documents, questions = load_fixture(Path("examples"))
    assert len(documents) == 2
    assert questions
    assert all(not Path(item["path"]).is_absolute() for item in documents)


def test_evaluation_runs_all_profiles_with_same_scope_and_budget(tmp_path):
    report = evaluate(
        FakeService(), Path("examples/eval_questions.json"), tmp_path / "report.json"
    )
    assert set(report["profiles"]) == {
        "vector",
        "vector_keyword",
        "vector_graph",
        "vector_topic_graph",
        "full",
    }
    for question in report["questions"]:
        scopes = {row["doc_scope"] for row in question["results"]}
        budgets = {row["top_k"] for row in question["results"]}
        assert len(scopes) == 1
        assert len(budgets) == 1
        assert all(row["retrieval_seconds"] >= 0 for row in question["results"])
        assert all("scores" in row for row in question["results"])


def test_markdown_report_keeps_auditable_fields(tmp_path):
    report = evaluate(FakeService(), Path("examples/eval_questions.json"))
    output = tmp_path / "report.md"

    write_markdown(report, output)

    markdown = output.read_text(encoding="utf-8")
    assert "Retrieval seconds" in markdown
    assert "Selected chunks" in markdown
    assert "doc::chunk_000000" in markdown
