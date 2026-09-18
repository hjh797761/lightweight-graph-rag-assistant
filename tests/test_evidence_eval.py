"""Real SQLite/service evaluator checks; model loaders are blocked from first RED."""
import copy
import importlib
import json

import pytest

from graphrag.config import AppConfig
from graphrag.embeddings import DeterministicEmbeddingBackend, EmbeddingBackend, CrossEncoderReranker
from graphrag.evidence_store import EvidenceStore
from graphrag.models import EvidenceOptions
from graphrag.service import GraphRAGService


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.setattr(EmbeddingBackend, "load", lambda *a, **k: pytest.fail("unexpected model load"))
    monkeypatch.setattr(CrossEncoderReranker, "load", lambda *a, **k: pytest.fail("unexpected reranker load"))
    monkeypatch.setattr(GraphRAGService, "_client", lambda *a: pytest.fail("answer API called"))


def module():
    assert importlib.util.find_spec("scripts.evidence_eval"), "evidence evaluator missing"
    return importlib.import_module("scripts.evidence_eval")


def dataset():
    return {"name": "original smoke", "purpose": "benchmark", "documents": [
        {"id": "a", "text": "# Section 1\nAlpha policy. See section 2.\n# Section 2\nException details.\n# Section 3\nOther."},
        {"id": "b", "text": "# Section 1\nIndependent source."}], "queries": [
        {"id": "q", "question": "Alpha policy", "split": "dev", "doc_scope": "a",
         "relevant_documents": {"a": 1}, "reference_target_ids": ["a::chunk_000001"],
         "reference_targets_exhaustive": True},
        {"id": "u", "question": "Unknown", "split": "dev"},
        {"id": "t", "question": "Independent", "split": "test", "relevant_documents": {"b": 2}}]}


def test_same_service_selection_reports_and_exclusive_output(tmp_path):
    opts = EvidenceOptions(max_chunks=2, max_supplements=1, context_budget=1800, supplement_fraction=.45)
    out = tmp_path / "run"
    report = module().run(dataset(), out, embedding_backend="deterministic", options=opts, split="dev")
    store = EvidenceStore(out / "evidence.sqlite3")
    cfg = AppConfig(kb_path=store.path, offline_mode=True, embedding_backend="deterministic",
                    enable_cross_encoder=False, evidence_options=opts)
    expected = GraphRAGService(cfg, store).retrieve("Alpha policy", doc_scope="a")
    row = report["queries"][0]
    assert row["chunk_ids"] == [c.id for c in expected.chunks]
    assert row["roles"] == expected.selection.roles
    assert row["context"] == expected.context
    assert row["budget_used"] == expected.selection.budget_used
    assert row["counts"] == expected.counts and row["counts"]["reranked_candidates"] == 0
    assert row["scores"] == expected.scores and row["candidate_details"] == expected.candidate_details
    assert row["decisions"] == expected.selection.decisions and row["reasons"] == expected.selection.reasons
    assert row["annotations"] == {k: dataset()["queries"][0][k] for k in (
        "relevant_documents", "reference_target_ids", "reference_targets_exhaustive")}
    assert report["queries"][1]["annotations"] == {}
    assert report["purpose"] == "smoke" and report["summary"]["query_count"] == 2
    assert report["summary"]["source_recall"]["denominator"] == 1
    assert report["summary"]["reference_coverage"]["denominator"] == 1
    assert report["queries"][1]["metrics"]["source_recall"] is None
    assert len(store.list_documents()) == 2
    persisted = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert persisted == report
    assert isinstance(persisted["runtime"]["embedding"], dict)
    assert all("embedding" not in detail for row in persisted["queries"] for detail in row["candidate_details"])
    assert expected.context in (out / "report.md").read_text(encoding="utf-8")
    before = (out / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        module().run(dataset(), out, embedding_backend="deterministic")
    assert (out / "report.json").read_bytes() == before


@pytest.mark.parametrize("mutation", [
    lambda d: d["documents"].append(copy.deepcopy(d["documents"][0])),
    lambda d: d["documents"][0].update(text="  "),
    lambda d: d["queries"][0].update(split="train"),
    lambda d: d["queries"][0].update(id=""),
    lambda d: d["queries"][0].update(relevant_documents={"missing": 1}),
    lambda d: d["queries"][2].update(relevant_documents={"a": 1}),
    lambda d: d["queries"][0].update(reference_target_ids=["missing"]),
    lambda d: d["queries"][0].update(reference_target_ids=["a::chunk_000001"] * 2),
    lambda d: d["queries"][0].update(reference_targets_exhaustive="true"),
    lambda d: d["queries"][0].pop("reference_target_ids"),
    lambda d: d["queries"][0].update(doc_scope="unknown-document"),
])
def test_invalid_input_precedes_output_and_model_load(tmp_path, mutation):
    data = dataset()
    mutation(data)
    with pytest.raises(ValueError):
        module().run(data, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_query_failure_keeps_denominator_and_id_and_cli_exit(tmp_path, monkeypatch):
    original = GraphRAGService.retrieve
    def fail(self, question, **kwargs):
        if question == "Alpha policy":
            raise RuntimeError("query failed")
        return original(self, question, **kwargs)
    monkeypatch.setattr(GraphRAGService, "retrieve", fail)
    path = tmp_path / "data.json"
    path.write_text(json.dumps(dataset()), encoding="utf-8")
    out = tmp_path / "run"
    assert module().main(["--dataset", str(path), "--out", str(out), "--embedding-backend", "deterministic"]) == 1
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["queries"][0]["id"] == "q" and "query failed" in report["queries"][0]["error"]
    assert report["summary"]["query_count"] == 2 and report["summary"]["failed_queries"] == 1
    assert report["summary"]["source_recall"] == {"value": 0., "denominator": 1, "status": "available"}
    assert report["summary"]["reference_coverage"]["value"] == 0.


def test_model_identity_offline_reranker_counts_and_prepared_sources(tmp_path, monkeypatch):
    calls, rerank_inputs = [], []
    out = tmp_path / "run"
    class Embedder(DeterministicEmbeddingBackend):
        def encode_many(self, texts, batch_size=20):
            assert len(EvidenceStore(out / "evidence.sqlite3").list_documents()) == 2
            return super().encode_many(texts, batch_size)
    def load(name, offline):
        calls.append((name, offline))
        return Embedder()
    def rerank(query, chunks):
        rerank_inputs.append(len(chunks))
        return [(c, float(len(chunks) - i)) for i, c in enumerate(chunks)]
    monkeypatch.setattr(EmbeddingBackend, "load", load)
    monkeypatch.setattr(CrossEncoderReranker, "load", lambda name, offline: (calls.append((name, offline)) or rerank))
    data = dataset()
    data["purpose"] = "smoke"
    report = module().run(data, out, model="D:/cached/model", rerank=True, reranker_model="D:/cached/reranker")
    assert calls == [("D:/cached/model", True), ("D:/cached/reranker", True)]
    assert report["purpose"] == "smoke" and report["configuration"]["rerank"] == "model"
    assert [q["counts"]["reranked_candidates"] for q in report["queries"]] == rerank_inputs
    EvidenceStore(out / "evidence.sqlite3").check_embedding_configuration("sentence_transformer", "D:/cached/model")


def test_manual_reference_labels_are_not_generated_and_exhaustive_is_required(tmp_path):
    data = dataset()
    data["queries"] = [{"id": "q", "question": "Alpha policy", "split": "dev", "doc_scope": "a"}]
    report = module().run(data, tmp_path / "unlabelled", embedding_backend="deterministic")
    for name in ("source_recall", "reference_coverage", "false_reference_expansion"):
        assert report["summary"][name] == {"value": None, "denominator": 0, "status": "unavailable"}
    data["queries"][0].update(reference_target_ids=[], reference_targets_exhaustive=True)
    # Inject ordering to ensure a real explicit reference supplement from the actual assembler.
    def reranker(question, chunks):
        return [(c, float(10 - c.sequence)) for c in chunks]
    report = module().run(data, tmp_path / "exhaustive", embedding_backend="deterministic", rerank=True,
                          reranker=reranker, options=EvidenceOptions(max_chunks=2, max_supplements=1, supplement_fraction=.45))
    assert report["configuration"]["rerank"] == "injected"
    assert report["summary"]["reference_coverage"]["value"] is None
    assert report["summary"]["false_reference_expansion"]["value"] == 1.


def test_token_count_requires_generator_counter_before_output(tmp_path):
    opts = EvidenceOptions(budget_unit="tokens")
    with pytest.raises(ValueError, match="count_tokens"):
        module().run(dataset(), tmp_path / "bad", embedding_backend="deterministic", options=opts)
    assert not (tmp_path / "bad").exists()
    report = module().run(dataset(), tmp_path / "ok", embedding_backend="deterministic", options=opts,
                          count_tokens=lambda text: len(text.split()))
    assert report["queries"][0]["budget_used"] == len(report["queries"][0]["context"].split())


def test_adjacent_supplement_not_false_reference_and_build_failure_has_no_report(tmp_path):
    data = dataset()
    data["documents"] = [{"id": "a", "text": "# Section 1\n" + "Alpha policy. " * 90}]
    data["queries"] = [{"id": "q", "question": "Alpha", "split": "dev",
                        "reference_target_ids": [], "reference_targets_exhaustive": True}]
    report = module().run(data, tmp_path / "adjacent", embedding_backend="deterministic", rerank=True,
        reranker=lambda q, chunks: [(c, float(10 - c.sequence)) for c in chunks],
        options=EvidenceOptions(max_chunks=2, max_supplements=1, supplement_fraction=.45))
    row = report["queries"][0]
    assert row["supplement_ids"] and all(r["kind"] == "adjacent" for id in row["supplement_ids"] for r in row["reasons"][id])
    assert report["summary"]["false_reference_expansion"]["denominator"] == 0
    class Broken:
        def encode_many(self, *args, **kwargs):
            raise RuntimeError("build failed")
    with pytest.raises(RuntimeError, match="build failed"):
        module().run(data, tmp_path / "broken", embedder=Broken())
    assert not (tmp_path / "broken" / "report.json").exists()


def test_cli_duplicate_relevance_ids_fail_before_output(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(dataset()).replace('"a": 1', '"a": 1, "a": 2'), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        module().main(["--dataset", str(path), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_inline_path_scope_and_exact_id_precedence(tmp_path):
    data = dataset()
    data["queries"] = [{"id": "path", "question": "Alpha", "split": "dev", "doc_scope": "inline:a"}]
    report = module().run(data, tmp_path / "path", embedding_backend="deterministic")
    assert report["queries"][0]["document_ids"] and set(report["queries"][0]["document_ids"]) == {"a"}
    data["documents"].append({"id": "inline:a", "text": "# Section 1\nAnother exact ID source."})
    report = module().run(data, tmp_path / "id", embedding_backend="deterministic")
    assert report["queries"][0]["document_ids"] == ["inline:a"]
