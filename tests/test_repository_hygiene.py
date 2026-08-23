from pathlib import Path


FORBIDDEN = ("中华第一库", "Super-N", "Fast-N", "650.33亿元", "60亿元股票回购")


def test_production_and_evaluator_contain_no_known_fixture_leaks():
    paths = list(Path("graphrag").glob("*.py")) + [Path("scripts/retrieval_eval.py")]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert not any(term in text for term in FORBIDDEN)


def test_readme_documents_real_entrypoints():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "OFFLINE_MODE=1" in readme
    assert "scripts/retrieval_eval.py" in readme
    assert "knowledge_base.sqlite3" in readme


def test_runtime_artifacts_are_ignored():
    ignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "*.sqlite3" in ignore
    assert ".tmp/" in ignore
