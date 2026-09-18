from dataclasses import replace
import sqlite3

import numpy as np
import pytest

from graphrag.embeddings import DeterministicEmbeddingBackend
from graphrag.models import ChunkRecord
from graphrag.retrieval import Retriever, vector_recall
from graphrag.storage import KnowledgeStore


class ObservedStore(KnowledgeStore):
    def __init__(self, path):
        super().__init__(path)
        self.reads = []

    def list_chunks(self, doc_id=None, chunk_ids=None):
        self.reads.append((doc_id, chunk_ids))
        return super().list_chunks(doc_id, chunk_ids)


def fixture(tmp_path):
    embedder = DeterministicEmbeddingBackend(32)
    records = [ChunkRecord(id=f"c{i:02d}", doc_id="doc", sequence=i,
                           clean_text=f"冷却 温度 控制 第{i}条", context_text=f"冷却 温度 控制 第{i}条",
                           concepts=("冷却", "控制"),
                           embedding=embedder.encode_one(f"冷却 温度 控制 第{i}条")) for i in range(12)]
    store = ObservedStore(tmp_path / "index.sqlite3")
    store.write_chunk_batch("doc", records, 12)
    return store, embedder, records


def test_full_retrieval_bounds_reranker_candidates_in_core(tmp_path):
    store, embedder, _ = fixture(tmp_path)
    calls = []

    def recording_reranker(query, candidates):
        calls.append([chunk.id for chunk in candidates])
        return [(chunk, float(len(candidates) - index))
                for index, chunk in enumerate(candidates)]

    result = Retriever(store, embedder, vector_recall_k=2,
                       reranker=recording_reranker).retrieve("冷却", profile="full", top_k=2)
    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert [chunk.id for chunk in result.chunks] == calls[0]


@pytest.mark.parametrize("candidate_k", [0, -1, 1.5, "2", None, True, False])
def test_retriever_rejects_invalid_candidate_limits(tmp_path, candidate_k):
    store, embedder, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="vector_recall_k.*positive integer"):
        Retriever(store, embedder, vector_recall_k=candidate_k)


def test_zero_keyword_hits_do_not_expand_vector_candidates(tmp_path):
    store, embedder, _ = fixture(tmp_path)
    retriever = Retriever(store, embedder, vector_recall_k=2)
    vector = retriever.retrieve("UNIQUE_TERM", profile="vector", top_k=12)
    keyword = retriever.retrieve("UNIQUE_TERM", profile="vector_keyword", top_k=12)
    assert len(keyword.chunks) == 2
    assert [(chunk.id, score) for chunk, score in zip(keyword.chunks, keyword.scores)] == [
        (chunk.id, score) for chunk, score in zip(vector.chunks, vector.scores)]


def test_disabled_reranking_preserves_positive_keyword_expansion_and_weights(tmp_path):
    store, embedder, records = fixture(tmp_path)
    ids, matrix = store.vector_matrix()
    vector_scores = dict(vector_recall(embedder.encode_one("冷却"), ids, matrix, 2))
    expected = sorted(((chunk.id, 0.70 * vector_scores.get(chunk.id, 0.0) + 0.20)
                       for chunk in records), key=lambda item: (-item[1], item[0]))
    result = Retriever(store, embedder, vector_recall_k=2).retrieve(
        "冷却", profile="vector_keyword", top_k=12)
    assert [(chunk.id, score) for chunk, score in zip(result.chunks, result.scores)] == expected


@pytest.mark.parametrize("candidate_k,expected_count", [(2, 2), (40, 12)])
def test_reranking_returns_only_available_candidates_when_top_k_is_larger(
        tmp_path, candidate_k, expected_count):
    store, embedder, _ = fixture(tmp_path)
    calls = []

    def recording_reranker(query, candidates):
        calls.append(len(candidates))
        return [(chunk, float(index)) for index, chunk in enumerate(reversed(candidates))]

    retriever = Retriever(store, embedder, vector_recall_k=candidate_k,
                          reranker=recording_reranker)
    result = retriever.retrieve("冷却", profile="full", top_k=20, fixed_output_k=20)
    assert calls == [expected_count]
    assert len(result.chunks) == expected_count
    assert retriever.retrieve("冷却", profile="full", doc_scope="missing", top_k=20).chunks == []
    assert calls == [expected_count]


def test_comparison_wrapper_reports_actual_reranker_input_without_truncating(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import external_comparison as comparison

    _, embedder, records = fixture(tmp_path)
    embedder.model = SimpleNamespace(device="cpu", max_seq_length=128,
                                    tokenizer=SimpleNamespace(encode=lambda text, **kwargs: list(text)))
    calls = []

    class RecordingReranker:
        model = SimpleNamespace(device="cpu")

        def __call__(self, query, candidates):
            calls.append(len(candidates))
            return [(chunk, float(index)) for index, chunk in enumerate(candidates)]

    def unbounded_adapter(chunks, embedder, output, profile, candidate_k, top_k, reranker, **kwargs):
        # An adapter regression must remain visible in the comparison diagnostics.
        return lambda query: reranker(query, chunks)

    monkeypatch.setattr(comparison.EmbeddingBackend, "load", lambda *args: embedder)
    monkeypatch.setattr(comparison.CrossEncoderReranker, "load", lambda *args: RecordingReranker())
    monkeypatch.setattr(comparison, "_project_retriever", unbounded_adapter)
    data = {"name": "unit", "purpose": "smoke",
            "documents": [{"id": chunk.id, "text": chunk.clean_text} for chunk in records[:3]],
            "queries": [{"id": "q", "question": "冷却", "split": "dev",
                         "relevant_documents": {"c00": 1}}]}
    report = comparison.run(data, tmp_path / "run", backend="model", rerank=True,
                            split="dev", candidate_k=2, top_k=2)
    assert calls == [3, 3]
    assert report["rows"][0]["reranked_candidates"] == 3


def test_warm_vector_only_reads_candidate_chunks_and_preserves_scores(tmp_path):
    store, embedder, _ = fixture(tmp_path)
    retriever = Retriever(store, embedder, vector_recall_k=5)
    ids, matrix = store.vector_matrix()
    expected = sorted(((i, 0.70 * s) for i, s in vector_recall(embedder.encode_one("冷却"), ids, matrix, 5)),
                      key=lambda x: (-x[1], x[0]))[:3]
    store.reads.clear()
    result = retriever.retrieve("冷却", profile="vector", top_k=3)
    assert store.reads and all(ids is not None and len(ids) <= 5 for _, ids in store.reads)
    assert [(c.id, s) for c, s in zip(result.chunks, result.scores)] == expected


def test_vector_skips_unused_concept_extraction(tmp_path, monkeypatch):
    import graphrag.retrieval as retrieval
    store, embedder, _ = fixture(tmp_path)
    def unused(_):
        pytest.fail("vector should not extract graph/keyword terms")
    monkeypatch.setattr(retrieval, "_terms", unused)
    assert Retriever(store, embedder).retrieve("冷却", profile="vector").chunks


def test_vector_updates_and_failed_transaction_preserve_consistency(tmp_path):
    store, embedder, records = fixture(tmp_path)
    retriever = Retriever(store, embedder)
    retriever.retrieve("新记录", profile="vector")
    updated = replace(records[0], clean_text="新记录", context_text="新记录",
                      embedding=embedder.encode_one("新记录"))
    store.write_chunk_batch("doc", [updated], 12)
    result = retriever.retrieve("新记录", profile="vector", top_k=12)
    assert result.chunks[0].clean_text == "新记录"
    before = [(c.id, s) for c, s in zip(result.chunks, result.scores)]
    with pytest.raises(sqlite3.IntegrityError):
        store.write_chunk_batch("doc", [replace(updated, clean_text="rolled back"),
                                         replace(updated, id="conflicting")], 13)
    after = retriever.retrieve("新记录", profile="vector", top_k=12)
    assert [(c.id, s) for c, s in zip(after.chunks, after.scores)] == before
    assert after.chunks[0].clean_text == "新记录"


def test_fixed_output_changes_only_final_selection_and_reports_stage_times(tmp_path):
    store, embedder, _ = fixture(tmp_path)
    calls = []
    def reranker(query, candidates):
        calls.append([c.id for c in candidates])
        return [(c, float(len(candidates) - i)) for i, c in enumerate(candidates)]
    retriever = Retriever(store, embedder, reranker=reranker)
    times = {}
    dynamic = retriever.retrieve("冷却", profile="full", top_k=10)
    fixed = retriever.retrieve("冷却", profile="full", top_k=10, fixed_output_k=10, stage_times=times)
    short = retriever.retrieve("冷却", profile="full", top_k=10, fixed_output_k=3)
    assert calls[0] == calls[1] == calls[2]
    assert len(dynamic.chunks) == len(short.chunks) == 3
    assert len(fixed.chunks) == 10
    assert [c.id for c in short.chunks] == [c.id for c in fixed.chunks[:3]]
    assert {"vector_data", "query_embedding", "chunk_data", "graph", "rerank", "selection_context"} <= times.keys()
    assert all(value >= 0 for value in times.values())
    with pytest.raises(ValueError, match="fixed_output_k"):
        retriever.retrieve("冷却", top_k=10, fixed_output_k=11)


@pytest.mark.parametrize("profile,graph,topics", [("vector", False, False), ("vector_keyword", False, False),
                                                   ("vector_graph", True, False), ("vector_topic_graph", True, True)])
def test_adapter_builds_only_profile_components(tmp_path, profile, graph, topics):
    from scripts.external_comparison import _project_retriever
    _, embedder, records = fixture(tmp_path)
    output = tmp_path / profile
    output.mkdir()
    _project_retriever(records, embedder, output, profile, 40, 10, None)
    store = KnowledgeStore(output / "index.sqlite3")
    assert bool(store.load_graph()) == graph
    assert bool(store.list_topics()) == topics
    assert any(c.concepts for c in store.list_chunks()) == graph


def test_runner_records_fixed_output_and_project_stage_diagnostics(tmp_path):
    from scripts.external_comparison import run
    _, _, records = fixture(tmp_path)
    data = {"name": "unit", "purpose": "smoke",
            "documents": [{"id": c.id, "text": c.clean_text} for c in records],
            "queries": [{"id": "q", "question": "冷却", "split": "dev", "relevant_documents": {"c00": 1}}]}
    report = run(data, tmp_path / "run", split="dev", profile="full", fixed_output_k=5)
    assert report["settings"]["dynamic_selection"] is False
    assert report["settings"]["fixed_output_k"] == 5
    assert report["summary"]["mean_returned_chunks"] == 5
    assert report["rows"][0]["retrieval_stage_seconds"]["graph"] >= 0
    assert report["summary"]["project_stage_p50_seconds"]["graph"] >= 0


@pytest.mark.parametrize("system", ["project", "llamaindex"])
def test_fixed_runner_output_is_prefix_of_unrestricted_output(tmp_path, system):
    if system == "llamaindex":
        pytest.importorskip("llama_index.core")
    from scripts.external_comparison import run
    _, _, records = fixture(tmp_path)
    data = {"name": "unit", "purpose": "smoke",
            "documents": [{"id": c.id, "text": c.clean_text} for c in records],
            "queries": [{"id": "q", "question": "冷却", "split": "dev", "relevant_documents": {"c00": 1}}]}
    whole = run(data, tmp_path / "whole", split="dev", system=system)
    short = run(data, tmp_path / "short", split="dev", system=system, fixed_output_k=3)
    assert short["rows"][0]["chunks"] == whole["rows"][0]["chunks"][:3]
    assert short["settings"]["top_k_chunk_budget"] == whole["settings"]["top_k_chunk_budget"] == 10
    if system == "llamaindex":
        assert short["rows"][0]["retrieval_stage_seconds"] is None


def test_candidate_fetch_preserves_scope_empty_results_and_ties(tmp_path):
    store, embedder, records = fixture(tmp_path)
    tied = [replace(records[0], id="z", sequence=0, doc_id="other"),
            replace(records[0], id="a", sequence=1, doc_id="other")]
    store.write_chunk_batch("other", tied, 2)
    retriever = Retriever(store, embedder)
    result = retriever.retrieve("冷却", profile="vector", doc_scope="other", top_k=10)
    assert [c.id for c in result.chunks] == ["a", "z"]
    assert retriever.retrieve("冷却", profile="vector", doc_scope="missing").chunks == []


def test_keyword_terms_normalized_once_per_query(tmp_path, monkeypatch):
    import graphrag.retrieval as retrieval
    store, embedder, _ = fixture(tmp_path)
    calls = []
    original = retrieval.normalize_evidence
    def observed(text):
        calls.append(text)
        return original(text)
    monkeypatch.setattr(retrieval, "normalize_evidence", observed)
    Retriever(store, embedder).retrieve("UNIQUE_TERM", profile="vector_keyword")
    assert calls.count("UNIQUE_TERM") <= 2


def test_exact_guard_without_numbers_does_not_normalize_corpus(tmp_path, monkeypatch):
    import graphrag.retrieval as retrieval
    store, embedder, _ = fixture(tmp_path)
    monkeypatch.setitem(retrieval.PROFILES, "exact_test",
                        replace(retrieval.PROFILES["vector"], name="exact_test", use_exact_guard=True))
    calls = []
    original = retrieval.normalize_evidence
    def observed(text):
        calls.append(text)
        return original(text)
    monkeypatch.setattr(retrieval, "normalize_evidence", observed)
    Retriever(store, embedder).retrieve("冷却", profile="exact_test")
    assert calls == ["冷却"]


def test_ablation_script_controls_process_seed_and_only_final_budget(tmp_path):
    import os
    from pathlib import Path
    import shutil
    import subprocess

    git = shutil.which("git")
    bash = Path(git).resolve().parents[1] / "bin" / "bash.exe" if os.name == "nt" and git else shutil.which("bash")
    if not bash or not Path(bash).exists():
        pytest.skip("Bash unavailable")
    data = tmp_path / "dataset.json"
    data.write_text("{}", encoding="utf-8")
    captured = tmp_path / "calls.txt"
    startup = tmp_path / "stubs.sh"
    startup.write_text('scontrol() { printf "test job\\n"; }\n'
                       'srun() { printf "seed=%s args=%s\\n" "${PYTHONHASHSEED-unset}" "$*" >> "$CAPTURE_CALLS"; }\n', encoding="utf-8")
    env = dict(os.environ, SLURM_JOB_ID="123", SLURM_SUBMIT_DIR=tmp_path.as_posix(),
               PYTHON_BIN="unused", HF_HOME=tmp_path.as_posix(), PILOT_DATASET=data.as_posix(),
               PYTHONHASHSEED="random", BASH_ENV=startup.as_posix(), CAPTURE_CALLS=captured.as_posix())
    script = Path(__file__).resolve().parents[1] / "cluster" / "crud_ablation.slurm"
    result = subprocess.run([str(bash), str(script)], env=env, capture_output=True)
    assert result.returncode == 0, result.stderr
    calls = captured.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 12
    assert all(line.startswith("seed=0 ") for line in calls)
    assert all("--top-k 10 --candidate-k 40" in line and "--require-cuda" in line for line in calls)
    assert sum("--fixed-output-k 3" in line for line in calls) == 3
    assert sum("--fixed-output-k 10" in line for line in calls) == 4
