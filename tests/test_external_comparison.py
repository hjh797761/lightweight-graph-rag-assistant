import math

import pytest


def test_ranking_metrics_use_all_relevant_documents_and_deduplicate():
    from scripts.comparison_metrics import ranking_metrics

    result = ranking_metrics(["b", "b", "a"], {"a": 3, "c": 1}, k=2)
    assert result["recall"] == 0.5
    assert result["mrr"] == 0.5
    assert result["ndcg"] == pytest.approx((7 / math.log2(3)) / (7 + 1 / math.log2(3)))


def test_empty_ranking_scores_zero_but_missing_labels_are_rejected():
    from scripts.comparison_metrics import ranking_metrics

    assert ranking_metrics([], {"a": 1}, k=10) == {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    with pytest.raises(ValueError, match="relevance"):
        ranking_metrics(["a"], {}, k=10)
    with pytest.raises(ValueError, match="positive"):
        ranking_metrics(["a"], {"a": 1}, k=0)


def dataset():
    return {
        "name": "unit-fixture",
        "purpose": "smoke",
        "documents": [{"id": "a", "text": "cooling valve"}, {"id": "b", "text": "revenue"}],
        "queries": [
            {"id": "q1", "question": "cooling?", "split": "dev", "relevant_documents": {"a": 1}},
            {"id": "q2", "question": "revenue?", "split": "test", "relevant_documents": {"b": 1}},
        ],
    }


def test_data_validation_rejects_unknown_targets_and_cross_split_leakage():
    from scripts.comparison_metrics import validate_dataset

    validate_dataset(dataset())
    bad = dataset()
    bad["queries"][1]["relevant_documents"] = {"missing": 1}
    with pytest.raises(ValueError, match="unknown"):
        validate_dataset(bad)
    bad["queries"][1]["relevant_documents"] = {"a": 1}
    with pytest.raises(ValueError, match="split"):
        validate_dataset(bad)


def test_duplicate_document_ids_are_rejected():
    from scripts.comparison_metrics import validate_dataset

    bad = dataset()
    bad["documents"].append(bad["documents"][0].copy())
    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset(bad)


def test_summary_counts_failed_queries_and_rejects_missing_queries():
    from scripts.comparison_metrics import summarize

    rows = [
        {"query_id": "q1", "document_ids": ["a"], "seconds": 1.0, "error": None},
        {"query_id": "q2", "document_ids": [], "seconds": 3.0, "error": "failure"},
    ]
    summary = summarize(dataset()["queries"], rows)
    assert summary["recall@10"] == 0.5
    assert summary["failed_queries"] == 1
    assert summary["queries"] == 2
    assert summary["p50_seconds"] == 2.0
    with pytest.raises(ValueError, match="query"):
        summarize(dataset()["queries"], rows[:1])


def test_project_adapter_returns_full_corpus_and_shared_source_ids(tmp_path):
    from scripts.external_comparison import run

    report = run(dataset(), tmp_path / "run", system="project", backend="deterministic", profile="vector", split="test")
    assert report["settings"]["scope"] == "whole_corpus"
    assert report["settings"]["reranker"] is None
    assert report["summary"]["queries"] == 1
    assert report["summary"]["failed_queries"] == 0
    assert set(report["rows"][0]["document_ids"]) == {"a", "b"}
    assert report["purpose"] == "smoke"
    assert (tmp_path / "run" / "report.md").exists()
    with pytest.raises(FileExistsError):
        run(dataset(), tmp_path / "run", system="project", backend="deterministic", profile="vector", split="test")


def test_real_llamaindex_adapter_uses_original_document_ids(tmp_path):
    pytest.importorskip("llama_index.core")
    from scripts.external_comparison import run

    report = run(dataset(), tmp_path / "llama", system="llamaindex", split="test")
    assert report["summary"]["failed_queries"] == 0
    assert set(report["rows"][0]["document_ids"]) == {"a", "b"}
    assert report["environment"]["llama-index-core"]
