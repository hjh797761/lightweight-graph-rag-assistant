"""Application-level evidence builds and compatibility, entirely local."""

from dataclasses import replace
import importlib
from pathlib import Path
import sys

import pytest

from graphrag.config import AppConfig
from graphrag.embeddings import DeterministicEmbeddingBackend
from graphrag.errors import ConfigurationError, IndexCapabilityError
from graphrag.evidence_store import EvidenceStore
from graphrag.ingestion import make_doc_id
from graphrag.models import EvidenceOptions
from graphrag.service import GraphRAGService, build_default_service
from graphrag.storage import KnowledgeStore


def config(path, **kwargs):
    return AppConfig(kb_path=path, offline_mode=True, embedding_backend="deterministic", enable_cross_encoder=False, **kwargs)


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    from graphrag.embeddings import EmbeddingBackend, CrossEncoderReranker
    monkeypatch.setattr(EmbeddingBackend, "load", lambda *a, **k: pytest.fail("unexpected model load"))
    monkeypatch.setattr(CrossEncoderReranker, "load", lambda *a, **k: pytest.fail("unexpected reranker load"))


def source(tmp_path, name="source.md"):
    path = tmp_path / name
    path.write_text("# Section 1\nSee section 2.\n# Section 2\nAn exception applies.\n# Section 3\nOther text.", encoding="utf-8")
    return path


def load_config(monkeypatch, value):
    monkeypatch.setattr(AppConfig, "load", classmethod(lambda cls, root: value))


def test_configuration_default_and_all_evidence_options(tmp_path, monkeypatch):
    assert AppConfig().retrieval_profile == "evidence"
    assert AppConfig().embedding_backend == "sentence_transformer"
    values = dict(RETRIEVAL_PROFILE="vector", EVIDENCE_CANDIDATE_K="7", EVIDENCE_LEXICAL_K="8",
                  EVIDENCE_RRF_K="9", EVIDENCE_MAX_CHUNKS="3", EVIDENCE_MAX_SUPPLEMENTS="1",
                  EVIDENCE_CONTEXT_BUDGET="900", EVIDENCE_SUPPLEMENT_FRACTION="0.2", EVIDENCE_BUDGET_UNIT="tokens")
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    loaded = AppConfig.load(tmp_path)
    assert loaded.retrieval_profile == "vector"
    assert loaded.evidence_options == EvidenceOptions(7, 8, 9, 3, 1, 900, .2, "tokens")


@pytest.mark.parametrize("name,value", [("RETRIEVAL_PROFILE", "oops"), ("EMBEDDING_BACKEND", "oops"),
    ("EVIDENCE_CONTEXT_BUDGET", "0"), ("EVIDENCE_MAX_SUPPLEMENTS", "-1"), ("EVIDENCE_CANDIDATE_K", "abc"),
    ("EVIDENCE_SUPPLEMENT_FRACTION", "nan"), ("EVIDENCE_BUDGET_UNIT", "bytes"), ("BUILD_BATCH_SIZE", "0")])
def test_bad_configuration_is_actionable(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError):
        AppConfig.load(tmp_path)


def test_default_creates_new_evidence_and_existing_open_is_readonly(tmp_path, monkeypatch):
    path = tmp_path / "new.sqlite3"
    load_config(monkeypatch, config(path))
    service = build_default_service(tmp_path)
    assert isinstance(service.store, EvidenceStore)
    before = path.read_bytes()
    assert isinstance(build_default_service(tmp_path).store, EvidenceStore)
    assert path.read_bytes() == before
    assert service.retrieve("empty").selection.status == "no_candidates"


def test_evidence_refuses_old_database_without_touching_it(tmp_path, monkeypatch):
    path = tmp_path / "old.sqlite3"
    KnowledgeStore(path)
    before = path.read_bytes()
    load_config(monkeypatch, config(path))
    with pytest.raises(IndexCapabilityError, match="build_evidence_index.*--out"):
        build_default_service(tmp_path)
    assert path.read_bytes() == before


def test_evidence_refuses_implicit_json_conversion(tmp_path, monkeypatch):
    path = tmp_path / "knowledge_base.json"
    path.write_text('{"documents": {}}', encoding="utf-8")
    before = path.read_bytes()
    load_config(monkeypatch, config(path))
    with pytest.raises(IndexCapabilityError, match="--out"):
        build_default_service(tmp_path)
    assert path.read_bytes() == before
    assert not path.with_suffix(".sqlite3").exists()


@pytest.mark.parametrize("profile", ["vector", "vector_keyword", "vector_graph", "vector_topic_graph", "full"])
def test_explicit_legacy_profile_keeps_old_database_bytes(tmp_path, monkeypatch, profile):
    path = tmp_path / "old.sqlite3"
    KnowledgeStore(path)
    before = path.read_bytes()
    load_config(monkeypatch, config(path, retrieval_profile=profile))
    service = build_default_service(tmp_path)
    assert service.retrieve("test").path.startswith(profile)
    assert path.read_bytes() == before


def test_explicit_legacy_json_migration_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / "old.json"
    path.write_bytes(Path("tests/fixtures/legacy_kb.json").read_bytes())
    before = path.read_bytes()
    load_config(monkeypatch, config(path, retrieval_profile="full"))
    assert build_default_service(tmp_path).store.count_chunks() == 2
    assert path.read_bytes() == before


def test_batch_resume_validates_entire_source_and_retrieval_families(tmp_path):
    path = source(tmp_path)
    store = EvidenceStore.create(tmp_path / "new.sqlite3")
    service = GraphRAGService(config(store.path), store)
    assert service.add_document(str(path), batch_size=1) == (1, 3)
    assert store.list_documents()[0]["state"] == "building"
    assert service.retrieve("exception").chunks == []
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace("Other text.", "Changed suffix."), encoding="utf-8")
    before = store.path.read_bytes()
    with pytest.raises(ValueError, match="source changed"):
        service.add_document(str(path), batch_size=1)
    assert store.path.read_bytes() == before
    path.write_text(original, encoding="utf-8")
    assert service.add_document(str(path), batch_size=5) == (3, 3)
    assert store.list_documents()[0]["state"] == "complete"
    assert service.retrieve("exception").selection is not None
    for profile in ("vector", "vector_keyword"):
        assert service.retrieve("exception", profile=profile).selection is None
    assert service.retrieve("exception").selection is not None
    for profile in ("vector_graph", "vector_topic_graph", "full"):
        with pytest.raises(IndexCapabilityError, match="graph|topic"):
            service.retrieve("exception", profile=profile)


def test_failed_embeddings_never_complete_document_and_changed_model_cannot_resume(tmp_path):
    class Broken:
        def encode_many(self, *args, **kwargs):
            raise RuntimeError("model failed")
    path = source(tmp_path)
    store = EvidenceStore.create(tmp_path / "new.sqlite3")
    service = GraphRAGService(config(store.path), store, embedder=Broken())
    with pytest.raises(RuntimeError, match="model failed"):
        service.add_document(str(path))
    assert store.list_documents()[0]["state"] == "building"
    assert store.count_chunks() == 0
    changed = GraphRAGService(replace(service.config, embedding_backend="sentence_transformer", embedding_model="other"), store, embedder=DeterministicEmbeddingBackend())
    before = store.path.read_bytes()
    with pytest.raises(ValueError, match="embedding configuration"):
        changed.add_document(str(path))
    assert store.path.read_bytes() == before


def test_public_three_tuple_uses_default_and_semantic_modes_remain_explicit(monkeypatch):
    module = importlib.import_module("graphrag_assistant")
    seen = []
    class Fake:
        def retrieve(self, query, doc_scope=None, profile=None):
            seen.append(profile)
            from graphrag.models import RetrievalResult
            return RetrievalResult(path="chosen", context="evidence")
    monkeypatch.setattr(module, "_service", Fake())
    assert module.retrieve("x") == ("chosen", "", "evidence")
    assert seen == [None]
    for mode, profile in (("embedding", "vector"), ("embedding_graph", "full"), ("evidence", "evidence"), ("vector_keyword", "vector_keyword"), ("vector_topic_graph", "vector_topic_graph")):
        module.retrieve_semantic("x", mode=mode)
        assert seen[-1] == profile
    with pytest.raises(ValueError):
        module.retrieve_semantic("x", mode="typo")


def test_reset_retains_store_type_and_exact_backup(tmp_path):
    store = EvidenceStore.create(tmp_path / "new.sqlite3")
    service = GraphRAGService(config(store.path), store)
    service.add_document(str(source(tmp_path)))
    before = store.path.read_bytes()
    backup = service.reset()
    assert backup.read_bytes() == before
    assert isinstance(service.store, EvidenceStore)
    assert service.list_documents() == []


def builder():
    assert importlib.util.find_spec("scripts.build_evidence_index") is not None, "explicit builder missing"
    return importlib.import_module("scripts.build_evidence_index")


def test_builder_exclusive_output_and_complete_resume_refusal(tmp_path, capsys):
    module = builder()
    path = source(tmp_path)
    out = tmp_path / "built.sqlite3"
    args = ["--input", str(path), "--out", str(out), "--embedding-backend", "deterministic", "--batch-size", "1"]
    assert module.main(args) == 0
    output = capsys.readouterr().out
    assert f"KB_PATH={out.resolve()}" in output
    assert "RETRIEVAL_PROFILE=evidence" in output and "EMBEDDING_BACKEND=deterministic" in output
    assert "EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5" in output and "OFFLINE_MODE=1" in output
    assert EvidenceStore(out).list_documents()[0]["state"] == "complete"
    before = out.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        module.main(args)
    with pytest.raises(ValueError, match="complete"):
        module.main(args + ["--resume"])
    assert out.read_bytes() == before


def test_builder_prepares_all_files_before_embedding_and_resume_validates_inputs(tmp_path, monkeypatch):
    module = builder()
    a, b = source(tmp_path, "a.md"), source(tmp_path, "b.md")
    out = tmp_path / "built.sqlite3"
    args = ["--input", str(a), str(b), "--out", str(out), "--embedding-backend", "deterministic", "--batch-size", "1"]
    original = DeterministicEmbeddingBackend.encode_many
    calls = []
    def interrupted(self, texts, batch_size):
        calls.append(texts)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return original(self, texts, batch_size)
    monkeypatch.setattr(DeterministicEmbeddingBackend, "encode_many", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        module.main(args)
    documents = EvidenceStore(out).list_documents()
    assert len(documents) == 2 and all(d["state"] == "building" for d in documents)
    assert sorted(d["processed_chunks"] for d in documents) == [0, 1]
    before = out.read_bytes()
    with pytest.raises(ValueError, match="input|pending"):
        module.main(["--input", str(a), "--out", str(out), "--resume", "--embedding-backend", "deterministic"])
    b.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        module.main(args + ["--resume"])
    assert out.read_bytes() == before
    source(tmp_path, "b.md")
    monkeypatch.setattr(DeterministicEmbeddingBackend, "encode_many", original)
    assert module.main(args + ["--resume"]) == 0
    assert all(d["state"] == "complete" for d in EvidenceStore(out).list_documents())


def test_builder_resume_refuses_legacy_database(tmp_path):
    module = builder()
    out = tmp_path / "old.sqlite3"
    KnowledgeStore(out)
    before = out.read_bytes()
    with pytest.raises(IndexCapabilityError):
        module.main(["--input", str(source(tmp_path)), "--out", str(out), "--resume", "--embedding-backend", "deterministic"])
    assert out.read_bytes() == before


def test_builder_rejects_json_output_that_application_would_redirect(tmp_path):
    out = tmp_path / "new.json"
    with pytest.raises(ValueError, match="JSON|json"):
        builder().main(["--input", str(source(tmp_path)), "--out", str(out), "--embedding-backend", "deterministic"])
    assert not out.exists() and not out.with_suffix(".sqlite3").exists()


def test_builder_model_default_is_cache_only_and_never_calls_answer_api(tmp_path, monkeypatch):
    module = builder()
    from graphrag.embeddings import EmbeddingBackend
    calls = []
    def load(name, offline):
        calls.append((name, offline))
        return DeterministicEmbeddingBackend()
    monkeypatch.setattr(EmbeddingBackend, "load", load)
    monkeypatch.setattr(GraphRAGService, "_client", lambda self: pytest.fail("answer/API client used"))
    path = source(tmp_path)
    for download in (False, True):
        args = ["--input", str(path), "--out", str(tmp_path / f"{download}.sqlite3"), "--model", "cached-model"]
        assert module.main(args + (["--allow-download"] if download else [])) == 0
        assert calls[-1] == ("cached-model", not download)


def test_pdf_parser_does_not_require_native_mupdf(tmp_path, monkeypatch):
    from test_structure import write_pdf_fixture
    path = tmp_path / "source.pdf"
    write_pdf_fixture(path, ["Evidence PDF."])
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    monkeypatch.setitem(sys.modules, "fitz", None)
    from graphrag.structure import parse_file
    assert "Evidence PDF" in parse_file(path, "doc")[0].source_text


def test_service_accepts_real_md_docx_pdf_and_keeps_source_locations(tmp_path):
    from zipfile import ZipFile
    from test_structure import write_pdf_fixture
    md = source(tmp_path)
    docx = tmp_path / "source.docx"
    with ZipFile(docx, "w") as package:
        package.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        package.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        package.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Word evidence.</w:t></w:r></w:p></w:body></w:document>')
    pdf = tmp_path / "source.pdf"
    write_pdf_fixture(pdf, ["PDF evidence."])
    store = EvidenceStore.create(tmp_path / "new.sqlite3")
    service = GraphRAGService(config(store.path), store)
    for path, locator in ((md, "lines"), (docx, "paragraphs"), (pdf, "page")):
        processed, total = service.add_document(str(path))
        assert processed == total and total > 0
        assert all(c.locator.startswith(locator) for c in store.list_chunks(make_doc_id(path)))
    assert all(d["state"] == "complete" for d in service.list_documents())


def test_token_budget_injection_and_provided_evidence_retriever(tmp_path):
    from graphrag.evidence_retrieval import EvidenceRetriever
    store = EvidenceStore.create(tmp_path / "new.sqlite3")
    cfg = config(store.path, evidence_options=EvidenceOptions(budget_unit="tokens", context_budget=100))
    count_tokens = lambda text: len(text.split())
    service = GraphRAGService(cfg, store, count_tokens=count_tokens)
    service.add_document(str(source(tmp_path)))
    result = service.retrieve("exception")
    assert result.selection.budget_unit == "tokens"
    assert result.selection.budget_used == count_tokens(result.context)
    retriever = EvidenceRetriever(store, DeterministicEmbeddingBackend(), options=cfg.evidence_options, count_tokens=count_tokens)
    injected = GraphRAGService(cfg, store, retriever=retriever)
    assert injected.retrieve("exception").context == result.context
    assert injected.retrieve("exception", profile="vector").selection is None
    assert injected.retrieve("exception").context == result.context


def test_same_dimension_different_model_is_rejected_before_query_or_resume(tmp_path):
    store = EvidenceStore.create(tmp_path / "new.sqlite3")
    cfg = config(store.path)
    service = GraphRAGService(cfg, store)
    path = source(tmp_path)
    service.add_document(str(path), batch_size=1)
    other = GraphRAGService(replace(cfg, embedding_model="different-same-dimension"), store)
    before = store.path.read_bytes()
    with pytest.raises(ValueError, match="embedding configuration"):
        other.add_document(str(path))
    with pytest.raises(ValueError, match="embedding configuration"):
        other.retrieve("exception")
    assert store.path.read_bytes() == before


def test_resume_validates_completed_inputs_and_skips_them(tmp_path, monkeypatch):
    module = builder()
    a, b = source(tmp_path, "a.md"), source(tmp_path, "b.md")
    store = EvidenceStore.create(tmp_path / "new.sqlite3")
    cfg = config(store.path)
    service = GraphRAGService(cfg, store)
    service.add_document(str(a))
    service.add_document(str(b), batch_size=1)
    args = ["--input", str(a), str(b), "--out", str(store.path), "--resume", "--embedding-backend", "deterministic"]
    before = store.path.read_bytes()
    a.write_text("Changed completed source", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        module.main(args)
    assert store.path.read_bytes() == before
    source(tmp_path, "a.md")
    original = DeterministicEmbeddingBackend.encode_many
    encoded = []
    def record(self, texts, batch_size):
        encoded.extend(texts)
        return original(self, texts, batch_size)
    monkeypatch.setattr(DeterministicEmbeddingBackend, "encode_many", record)
    assert module.main(args) == 0
    assert len(encoded) == 2


def test_menu_prints_readable_selection_limits_and_retains_reset_confirmation(tmp_path, monkeypatch, capsys):
    from graphrag.models import EvidenceChunk, EvidenceSelection, RetrievalResult
    module = importlib.import_module("graphrag_assistant")
    chunk = EvidenceChunk("c", "d", 0, "text", "text", locator="lines 1-1", partial=True)
    selection = EvidenceSelection(chunks=[chunk], roles=["supplement"], budget_used=15,
        reasons={"c": [{"kind": "explicit_reference", "basis": "section 2", "source_id": "original"}]},
        status="ok", decisions=[{"phase": "association", "status": "no_links"}, {"status": "partial_coverage", "source_id": "original"}])
    result = RetrievalResult(path="evidence", chunks=[chunk], selection=selection)
    module._print_retrieval(result)
    output = capsys.readouterr().out
    assert "补充证据" in output and "section 2" in output and "15 characters" in output
    assert "无可用关联" in output and "部分覆盖" in output and "原文片段不完整" in output
    class Fake:
        def load(self): return {}
        def list_documents(self): return []
        def reset(self, **kwargs): pytest.fail("reset without exact confirmation")
    monkeypatch.setattr(module, "_service", Fake())
    answers = iter(["9", "yes", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    module.main()
    assert "轻量证据关联助手" in capsys.readouterr().out
