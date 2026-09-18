import copy

import pytest


def source_fixture(count=6):
    data = {"questanswer_1doc": [], "questanswer_2docs": [], "questanswer_3docs": []}
    for event in range(count):
        for n, task in enumerate(data, 1):
            row = {"ID": f"event-{event}", "questions": f"question {event} {n}",
                   "answers": "DO NOT INDEX ANSWER", "thoughts": "DO NOT INDEX THOUGHT"}
            row.update({f"news{i}": f"article {event} {i}" for i in range(1, n + 1)})
            data[task].append(row)
    return data


def test_pilot_groups_queries_and_includes_distractors_without_answer_leakage():
    from scripts.prepare_crud_pilot import prepare

    data, audit = prepare(source_fixture(), query_events=2, corpus_events=4, seed=7)
    assert len(data["queries"]) == 6
    assert len(data["documents"]) == 12
    assert {q["split"] for q in data["queries"]} == {"dev"}
    assert len({q["event_id"] for q in data["queries"]}) == 2
    assert all(len([q for q in data["queries"] if q["event_id"] == event]) == 3
               for event in audit["query_event_ids"])
    assert all("DO NOT INDEX" not in doc["text"] for doc in data["documents"])
    assert all("answers" not in q and "thoughts" not in q for q in data["queries"])
    assert data["metadata"]["relevance_kind"] == "source_document_proxy"
    assert prepare(source_fixture(), query_events=2, corpus_events=4, seed=7) == (data, audit)


def test_pilot_audits_missing_sources_and_deduplicates_text():
    from scripts.prepare_crud_pilot import prepare

    source = source_fixture()
    source["questanswer_3docs"][0]["news3"] = "  "
    source["questanswer_2docs"][1]["news2"] = " article 1 1 "
    data, audit = prepare(source, query_events=5, corpus_events=5, seed=7)
    assert "event-0" not in audit["query_event_ids"]
    assert audit["excluded_rows"][0]["reason"] == "missing_or_empty_news3"
    assert len({d["text"] for d in data["documents"]}) == len(data["documents"])
    q = next(q for q in data["queries"] if q["event_id"] == "event-1" and q["source_task"] == "questanswer_2docs")
    assert len(q["relevant_documents"]) == 1


def test_pilot_rejects_invalid_counts_and_duplicate_source_ids():
    from scripts.prepare_crud_pilot import prepare

    for queries, corpus in [(0, 4), (3, 2), (2, 99)]:
        with pytest.raises(ValueError):
            prepare(source_fixture(), query_events=queries, corpus_events=corpus, seed=7)
    data = source_fixture()
    data["questanswer_1doc"].append(copy.deepcopy(data["questanswer_1doc"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        prepare(data, query_events=2, corpus_events=4, seed=7)


def test_source_proxy_report_preserves_metadata_and_names_metrics(tmp_path):
    from scripts.external_comparison import run
    from scripts.prepare_crud_pilot import prepare

    data, _ = prepare(source_fixture(), query_events=2, corpus_events=4, seed=7)
    report = run(data, tmp_path / "proxy", split="dev")
    assert report["dataset_metadata"] == data["metadata"]
    assert "source_recall@10" in report["summary"]
    assert "recall@10" not in report["summary"]
    assert report["summary"]["mean_returned_context_characters"] > 0
    assert report["summary"]["mean_returned_context_embedding_tokens"] is None
    assert "not exhaustive relevance" in (tmp_path / "proxy" / "report.md").read_text(encoding="utf-8")


def test_required_cuda_rejects_cpu_smoke(tmp_path):
    from scripts.external_comparison import run
    from scripts.prepare_crud_pilot import prepare

    data, _ = prepare(source_fixture(), query_events=2, corpus_events=4, seed=7)
    with pytest.raises(ValueError, match="CUDA"):
        run(data, tmp_path / "cpu", split="dev", require_cuda=True)


def test_slurm_reused_job_id_preserves_existing_metadata(tmp_path):
    import os
    from pathlib import Path
    import shutil
    import subprocess

    git = shutil.which("git")
    bash = Path(git).resolve().parents[1] / "bin" / "bash.exe" if os.name == "nt" and git else shutil.which("bash")
    if not bash or not Path(bash).exists():
        pytest.skip("Bash unavailable")
    root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / ".tmp" / "cluster" / "crud-pilot-123"
    run_dir.mkdir(parents=True)
    marker = run_dir / "slurm-start.txt"
    marker.write_text("preserve previous job metadata", encoding="utf-8")
    data = tmp_path / "dataset.json"
    data.write_text("{}", encoding="utf-8")
    env = dict(os.environ, SLURM_JOB_ID="123", SLURM_SUBMIT_DIR=tmp_path.as_posix(),
               PYTHON_BIN="unused", HF_HOME=tmp_path.as_posix(), PILOT_DATASET=data.as_posix())
    result = subprocess.run([str(bash), str(root / "cluster" / "crud_pilot.slurm")],
                            env=env, capture_output=True)
    assert result.returncode != 0
    assert marker.read_text(encoding="utf-8") == "preserve previous job metadata"
