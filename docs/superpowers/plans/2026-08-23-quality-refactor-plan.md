# Comprehensive Quality Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic JSON-backed prototype with a compatible, CPU-first package using SQLite, reproducible same-pipeline evaluation, generic prompts, batch embeddings, tests, and CI.

**Architecture:** Keep `graphrag_assistant.py` as a compatibility/CLI shell and move responsibilities into the `graphrag` package. Persist batches in SQLite, expose embeddings as cached NumPy matrices, and run all ablations through one retrieval pipeline configured by `RetrievalProfile`.

**Tech Stack:** Python 3.10+, SQLite, NumPy, sentence-transformers, python-dotenv, pytest, GitHub Actions.

---

## File map

- `graphrag/config.py`: typed environment configuration and `.env` loading.
- `graphrag/errors.py`: explicit domain exceptions.
- `graphrag/models.py`: dataclasses shared across modules.
- `graphrag/normalization.py`: generic text/evidence normalization.
- `graphrag/embeddings.py`: online/offline model creation and batch encoding.
- `graphrag/storage.py`: SQLite schema, batch writes, reads, vector cache.
- `graphrag/migration.py`: non-destructive legacy JSON migration.
- `graphrag/ingestion.py`: current-format readers, splitting, batch ingestion.
- `graphrag/graph.py`: current local concept/topic behavior behind explicit interfaces.
- `graphrag/retrieval.py`: same-pipeline profiles, recall, rerank, context construction.
- `graphrag/answering.py`: provider call and generic evidence-grounded prompt.
- `graphrag/service.py`: application orchestration.
- `graphrag_assistant.py`: compatible public functions and CLI.
- `scripts/retrieval_eval.py`: public evaluation runner.
- `scripts/cpu_benchmark.py`: non-gating CPU timings.
- `examples/eval_*.txt`, `examples/eval_questions.json`: lawful synthetic fixtures.
- `tests/`: unit, migration, integration, compatibility, and evaluation tests.
- `.github/workflows/tests.yml`: Windows/Linux CI.

### Task 1: Package foundation and configuration

**Files:**
- Create: `graphrag/__init__.py`
- Create: `graphrag/errors.py`
- Create: `graphrag/models.py`
- Create: `graphrag/config.py`
- Create: `tests/test_config.py`
- Modify: `.env.example`
- Modify: `requirements.txt`
- Create: `requirements-eval.txt`
- Create: `requirements-dev.txt`

- [ ] **Step 1: Write failing configuration tests**

```python
from pathlib import Path

from graphrag.config import AppConfig


def test_environment_overrides_dotenv(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text("OFFLINE_MODE=1\nVECTOR_RECALL_K=7\n", encoding="utf-8")
    monkeypatch.setenv("VECTOR_RECALL_K", "11")
    config = AppConfig.load(project_root=tmp_path)
    assert config.offline_mode is True
    assert config.vector_recall_k == 11


def test_json_kb_path_derives_sqlite_path(tmp_path: Path):
    config = AppConfig(kb_path=tmp_path / "knowledge_base.json")
    assert config.database_path == tmp_path / "knowledge_base.sqlite3"
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL because `graphrag.config` does not exist.

- [ ] **Step 3: Implement errors, shared models, and typed config**

```python
# graphrag/errors.py
class GraphRAGError(Exception):
    pass

class ConfigurationError(GraphRAGError):
    pass

class ModelUnavailableError(GraphRAGError):
    pass

class DocumentParseError(GraphRAGError):
    pass

class StorageError(GraphRAGError):
    pass

class MigrationError(GraphRAGError):
    pass
```

```python
# graphrag/config.py
from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import dotenv_values

@dataclass(frozen=True)
class AppConfig:
    kb_path: Path = Path("knowledge_base.json")
    offline_mode: bool = False
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    cross_encoder_model: str = "BAAI/bge-reranker-base"
    enable_cross_encoder: bool = True
    vector_recall_k: int = 40
    final_top_k: int = 6
    batch_size: int = 20

    @property
    def database_path(self) -> Path:
        return self.kb_path.with_suffix(".sqlite3") if self.kb_path.suffix.lower() == ".json" else self.kb_path

    @classmethod
    def load(cls, project_root: Path) -> "AppConfig":
        file_values = dotenv_values(project_root / ".env")
        values = {**file_values, **os.environ}
        return cls(
            kb_path=Path(values.get("KB_PATH") or project_root / "knowledge_base.json"),
            offline_mode=str(values.get("OFFLINE_MODE", "0")) == "1",
            embedding_model=str(values.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")),
            cross_encoder_model=str(values.get("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")),
            enable_cross_encoder=str(values.get("ENABLE_CROSS_ENCODER", "1")) == "1",
            vector_recall_k=int(values.get("VECTOR_RECALL_K", "40")),
            final_top_k=int(values.get("FINAL_TOP_K", "6")),
            batch_size=int(values.get("BUILD_BATCH_SIZE", "20")),
        )
```

```python
# graphrag/models.py
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class ChunkRecord:
    id: str
    doc_id: str
    sequence: int
    clean_text: str
    context_text: str
    chapter: str = "未分类"
    concepts: tuple[str, ...] = ()
    embedding: Any = None

@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    use_keyword: bool
    use_graph: bool
    use_topic: bool
    use_exact_guard: bool
    use_bridges: bool
    dynamic_top_k: bool

@dataclass
class RetrievalResult:
    path: str
    chunks: list[ChunkRecord] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    context: str = ""
```

- [ ] **Step 4: Split dependencies and document offline mode**

```text
# requirements-eval.txt
deepeval>=3.0.0
```

```text
# requirements-dev.txt
numpy>=1.24.0
python-dotenv>=1.0.0
pytest>=8.0.0
```

Add `python-dotenv>=1.0.0` to `requirements.txt`, remove `deepeval`, and add `OFFLINE_MODE=0` plus `BUILD_BATCH_SIZE=20` to `.env.example`.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_config.py -v`

Expected: 2 passed.

```bash
git add graphrag tests/test_config.py .env.example requirements.txt requirements-eval.txt requirements-dev.txt
git commit -m "refactor: add package configuration foundation"
```

### Task 2: Generic normalization and answer prompting

**Files:**
- Create: `graphrag/normalization.py`
- Create: `graphrag/answering.py`
- Create: `tests/test_normalization.py`
- Create: `tests/test_answering.py`

- [ ] **Step 1: Write failing genericity tests**

```python
from graphrag.answering import build_answer_messages
from graphrag.normalization import normalize_evidence

CONTAMINATED = ("中华第一库", "Super-N", "Fast-N", "650.33", "60亿元股票回购")

def test_normalization_is_generic():
    assert normalize_evidence("收入 1,234.50 万元，同比增长 8 %") == "收入1234.50万元同比增长8%"

def test_prompt_contains_no_private_fixture_terms():
    rendered = repr(build_answer_messages("问题", "[S1] 资料"))
    assert not any(term in rendered for term in CONTAMINATED)

def test_context_is_passed_with_source_labels():
    messages = build_answer_messages("指标是多少？", "[S1] 指标为 8%。")
    assert messages[0]["role"] == "system"
    assert "[S1]" in messages[1]["content"]
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_normalization.py tests/test_answering.py -v`

Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement domain-independent normalization and prompt**

```python
# graphrag/normalization.py
import re
import unicodedata

def normalize_evidence(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    value = re.sub(r"[\s,.;:，。；：]+", "", value)
    return value
```

```python
# graphrag/answering.py
def build_answer_messages(query: str, context: str) -> list[dict[str, str]]:
    system = (
        "你是基于证据的问答助手。只使用带来源编号的资料。"
        "先给结论，再覆盖问题中的主体、时间、指标、单位、原因和条件。"
        "资料不足时明确写资料未提供。每个事实后标注对应的 [S编号]。"
    )
    user = f"用户问题：\n{query}\n\n证据资料：\n{context}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def generate_answer(client, model: str, query: str, context: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=build_answer_messages(query, context),
        temperature=0,
    )
    return completion.choices[0].message.content.strip()
```

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_normalization.py tests/test_answering.py -v`

Expected: 3 passed.

```bash
git add graphrag/normalization.py graphrag/answering.py tests/test_normalization.py tests/test_answering.py
git commit -m "refactor: replace fixture-specific answer rules"
```

### Task 3: Online/offline models and batch embeddings

**Files:**
- Create: `graphrag/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing model and batch tests**

```python
import numpy as np
from graphrag.embeddings import EmbeddingBackend

class FakeModel:
    def __init__(self):
        self.calls = []
    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.asarray([[len(text), 1.0] for text in texts], dtype=np.float64)

def test_batch_encode_returns_float32_matrix():
    fake = FakeModel()
    backend = EmbeddingBackend(model=fake)
    result = backend.encode_many(["甲", "乙乙"], batch_size=2)
    assert result.dtype == np.float32
    assert result.shape == (2, 2)
    assert len(fake.calls) == 1

def test_loader_receives_offline_flag(monkeypatch):
    seen = {}
    def loader(name, **kwargs):
        seen.update(name=name, **kwargs)
        return FakeModel()
    EmbeddingBackend.load("model-name", offline=True, loader=loader)
    assert seen == {"name": "model-name", "local_files_only": True}
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_embeddings.py -v`

Expected: FAIL because `EmbeddingBackend` is missing.

- [ ] **Step 3: Implement lazy loading and batch encoding**

```python
import numpy as np
from .errors import ModelUnavailableError

class EmbeddingBackend:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, name: str, offline: bool, loader=None):
        if loader is None:
            from sentence_transformers import SentenceTransformer
            loader = SentenceTransformer
        try:
            return cls(loader(name, local_files_only=offline))
        except Exception as exc:
            mode = "离线缓存" if offline else "在线下载"
            raise ModelUnavailableError(f"无法通过{mode}加载模型 {name}") from exc

    def encode_many(self, texts: list[str], batch_size: int) -> np.ndarray:
        values = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(values, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode_many([text], batch_size=1)[0]
```

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_embeddings.py -v`

Expected: 2 passed.

```bash
git add graphrag/embeddings.py tests/test_embeddings.py
git commit -m "perf: add batch embedding backend"
```

### Task 4: SQLite storage and vector cache

**Files:**
- Create: `graphrag/storage.py`
- Create: `tests/test_storage.py`

- [ ] **Step 1: Write failing atomicity and cache tests**

```python
import numpy as np
import pytest
from graphrag.models import ChunkRecord
from graphrag.storage import KnowledgeStore

def chunk(seq: int) -> ChunkRecord:
    return ChunkRecord(
        id=f"doc::chunk_{seq:06d}", doc_id="doc", sequence=seq,
        clean_text=f"text {seq}", context_text=f"text {seq}",
        concepts=("概念",), embedding=np.asarray([seq + 1.0, 1.0], dtype=np.float32),
    )

def test_batch_failure_rolls_back_chunks_and_progress(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.sqlite3")
    with pytest.raises(RuntimeError):
        store.write_chunk_batch("doc", [chunk(0), chunk(1)], next_index=2, fail_after_chunks=True)
    assert store.count_chunks() == 0
    assert store.get_progress("doc") == 0

def test_vector_cache_is_invalidated_after_write(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.sqlite3")
    store.write_chunk_batch("doc", [chunk(0)], next_index=1)
    ids1, matrix1 = store.vector_matrix("doc")
    store.write_chunk_batch("doc", [chunk(1)], next_index=2)
    ids2, matrix2 = store.vector_matrix("doc")
    assert ids1 == ["doc::chunk_000000"]
    assert ids2 == ["doc::chunk_000000", "doc::chunk_000001"]
    assert matrix1.shape == (1, 2) and matrix2.shape == (2, 2)
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_storage.py -v`

Expected: FAIL because `KnowledgeStore` does not exist.

- [ ] **Step 3: Implement schema, transactional batch writes, and cache**

```python
SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(
  id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '', path TEXT NOT NULL DEFAULT '',
  processed_chunks INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks(
  id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, sequence INTEGER NOT NULL,
  clean_text TEXT NOT NULL, context_text TEXT NOT NULL, chapter TEXT NOT NULL,
  embedding BLOB NOT NULL, embedding_dim INTEGER NOT NULL,
  UNIQUE(doc_id, sequence)
);
CREATE TABLE IF NOT EXISTS chunk_concepts(
  chunk_id TEXT NOT NULL, concept TEXT NOT NULL, PRIMARY KEY(chunk_id, concept)
);
CREATE TABLE IF NOT EXISTS concept_edges(
  source TEXT NOT NULL, target TEXT NOT NULL, weight INTEGER NOT NULL,
  PRIMARY KEY(source, target)
);
CREATE TABLE IF NOT EXISTS topics(
  id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, title TEXT NOT NULL,
  centroid BLOB NOT NULL, embedding_dim INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS topic_chunks(
  topic_id TEXT NOT NULL, chunk_id TEXT NOT NULL, PRIMARY KEY(topic_id, chunk_id)
);
"""
```

Implement `KnowledgeStore.__init__`, `write_chunk_batch`, `count_chunks`, `get_progress`, `vector_matrix`, `list_chunks`, `replace_topics`, and `load_graph`. Use `with connection:` around chunk rows, concept rows, edge increments, and progress update. Convert vectors with `np.asarray(vector, dtype=np.float32).tobytes()`. Clear `self._vector_cache` only after a successful commit.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_storage.py -v`

Expected: 2 passed.

```bash
git add graphrag/storage.py tests/test_storage.py
git commit -m "refactor: add atomic SQLite knowledge store"
```

### Task 5: Non-destructive JSON migration

**Files:**
- Create: `graphrag/migration.py`
- Create: `tests/fixtures/legacy_kb.json`
- Create: `tests/test_migration.py`

- [ ] **Step 1: Write failing migration tests**

```python
import json
from pathlib import Path
import pytest
from graphrag.migration import migrate_legacy_json
from graphrag.storage import KnowledgeStore

def test_migration_preserves_json_and_counts(tmp_path: Path):
    source = tmp_path / "knowledge_base.json"
    source.write_text(Path("tests/fixtures/legacy_kb.json").read_text(encoding="utf-8"), encoding="utf-8")
    target = migrate_legacy_json(source)
    assert source.exists()
    assert target == tmp_path / "knowledge_base.sqlite3"
    assert KnowledgeStore(target).count_chunks() == 2

def test_failed_migration_does_not_replace_target(tmp_path: Path):
    source = tmp_path / "knowledge_base.json"
    source.write_text("{broken", encoding="utf-8")
    with pytest.raises(Exception):
        migrate_legacy_json(source)
    assert source.read_text(encoding="utf-8") == "{broken"
    assert not (tmp_path / "knowledge_base.sqlite3").exists()
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_migration.py -v`

Expected: FAIL because migration code is missing.

- [ ] **Step 3: Implement temporary-database migration**

```python
def migrate_legacy_json(source: Path) -> Path:
    target = source.with_suffix(".sqlite3")
    temporary = target.with_suffix(".sqlite3.migrating")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
        store = KnowledgeStore(temporary)
        store.import_legacy(data)
        expected = len(data.get("chunks", {}))
        if store.count_chunks() != expected:
            raise MigrationError(f"迁移计数不一致: {store.count_chunks()} != {expected}")
        os.replace(temporary, target)
        return target
    except Exception as exc:
        raise MigrationError(f"旧知识库迁移失败，原文件保留于 {source}") from exc
```

The fixture must contain two documents/chunks, embeddings, concepts, graph, topic, and processed progress matching schema version 3.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_migration.py -v`

Expected: 2 passed.

```bash
git add graphrag/migration.py tests/fixtures/legacy_kb.json tests/test_migration.py
git commit -m "feat: migrate legacy JSON knowledge bases"
```

### Task 6: Ingestion, graph, and topic extraction

**Files:**
- Create: `graphrag/ingestion.py`
- Create: `graphrag/graph.py`
- Create: `tests/test_ingestion.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write failing batch-ingestion and graph tests**

```python
def test_ingestion_encodes_one_batch_and_saves_one_batch(tmp_path):
    embedder = RecordingEmbedder()
    store = RecordingStore()
    ingest_text("doc", "第一节\n甲。\n\n第二节\n乙。", store, embedder, batch_size=20)
    assert embedder.batch_calls == 1
    assert store.batch_calls == 1

def test_concept_edges_are_generic_and_symmetric():
    graph = build_concept_edges([("甲概念", "乙概念", "甲概念")])
    assert graph[("甲概念", "乙概念")] == 1
    assert graph[("乙概念", "甲概念")] == 1
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_ingestion.py tests/test_graph.py -v`

Expected: FAIL because ingestion and graph modules are missing.

- [ ] **Step 3: Move current generic logic behind interfaces**

Move `load_pdf`, `load_text`, `split_text`, noise filtering, local chapter choice, local concept extraction, topic clustering, and graph edge construction from `graphrag_assistant.py`. Preserve current algorithms except removal of fixture-specific signal words.

```python
def ingest_text(doc_id, text, store, embedder, batch_size):
    pairs = split_text(text)
    start = store.get_progress(doc_id)
    selected = pairs[start:start + batch_size]
    vectors = embedder.encode_many([clean for clean, _ in selected], batch_size=batch_size)
    chunks = [
        make_chunk_record(doc_id, start + offset, clean, context, vectors[offset])
        for offset, (clean, context) in enumerate(selected)
    ]
    store.write_chunk_batch(doc_id, chunks, next_index=start + len(chunks))
    return len(chunks), len(pairs)
```

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_ingestion.py tests/test_graph.py -v`

Expected: all tests pass and no test sleeps.

```bash
git add graphrag/ingestion.py graphrag/graph.py tests/test_ingestion.py tests/test_graph.py
git commit -m "refactor: extract batch ingestion and graph indexing"
```

### Task 7: Same-pipeline retrieval profiles

**Files:**
- Create: `graphrag/retrieval.py`
- Create: `tests/test_retrieval_profiles.py`

- [ ] **Step 1: Write failing profile and matrix-recall tests**

```python
import numpy as np
import pytest
from graphrag.retrieval import PROFILES, vector_recall

def test_profiles_are_explicit_ablations():
    assert PROFILES["vector"].use_graph is False
    assert PROFILES["vector_graph"].use_graph is True
    assert PROFILES["full"].use_bridges is True

def test_vector_recall_uses_matrix_scores():
    ids = ["a", "b"]
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    hits = vector_recall(np.asarray([0.9, 0.1], dtype=np.float32), ids, matrix, top_k=1)
    assert hits == [("a", pytest.approx(0.9))]

def test_every_profile_preserves_scope_and_budget(fake_pipeline):
    results = [fake_pipeline.retrieve("q", profile=name, doc_scope="doc", top_k=4) for name in PROFILES]
    assert all(result.requested_scope == "doc" for result in results)
    assert all(result.requested_top_k == 4 for result in results)
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_retrieval_profiles.py -v`

Expected: FAIL because retrieval module is missing.

- [ ] **Step 3: Extract retrieval stages and add profiles**

```python
PROFILES = {
    "vector": RetrievalProfile("vector", False, False, False, False, False, False),
    "vector_keyword": RetrievalProfile("vector_keyword", True, False, False, False, False, False),
    "vector_graph": RetrievalProfile("vector_graph", False, True, False, False, False, False),
    "vector_topic_graph": RetrievalProfile("vector_topic_graph", False, True, True, False, False, False),
    "full": RetrievalProfile("full", True, True, True, True, True, True),
}

def vector_recall(query_vector, ids, matrix, top_k):
    scores = matrix @ query_vector
    order = np.argsort(scores)[::-1][:top_k]
    return [(ids[index], float(scores[index])) for index in order]
```

Move current generic query terms, graph expansion, topic routing, rerank, cross-encoder, dynamic selection, bridges, and context building into a `Retriever` class. Keep existing weights and defaults. Context blocks must be labeled `[S1]`, `[S2]` and include document, page text when present, and chunk ID.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_retrieval_profiles.py -v`

Expected: all profile, scope, budget, and vector tests pass.

```bash
git add graphrag/retrieval.py tests/test_retrieval_profiles.py
git commit -m "refactor: unify retrieval and ablation profiles"
```

### Task 8: Application service and compatibility shell

**Files:**
- Create: `graphrag/service.py`
- Replace: `graphrag_assistant.py`
- Modify: `dingtalk_server.py`
- Create: `tests/test_compatibility.py`

- [ ] **Step 1: Write failing compatibility tests**

```python
import graphrag_assistant as legacy

def test_legacy_public_functions_exist():
    for name in ("load_knowledge_base", "retrieve", "retrieve_semantic", "generate_answer", "add_document_to_tree"):
        assert callable(getattr(legacy, name))

def test_legacy_retrieve_returns_three_values(monkeypatch):
    monkeypatch.setattr(legacy, "_service", FakeService())
    path, selected, context = legacy.retrieve("问题")
    assert path == "full召回-top-1 -> 选出-top-1"
    assert selected == "doc::chunk_000000"
    assert "[S1]" in context
```

- [ ] **Step 2: Verify the tests fail after moving implementation**

Run: `python -m pytest tests/test_compatibility.py -v`

Expected: FAIL until the compatibility shell is added.

- [ ] **Step 3: Implement service and wrapper**

```python
# graphrag/service.py
class GraphRAGService:
    def __init__(self, config, store, embedder, retriever, llm_client_factory):
        self.config = config
        self.store = store
        self.embedder = embedder
        self.retriever = retriever
        self.llm_client_factory = llm_client_factory

    def retrieve(self, query, doc_scope=None, profile="full"):
        return self.retriever.retrieve(query, doc_scope=doc_scope, profile=profile)
```

```python
# graphrag_assistant.py public shell
_service = build_default_service()

def load_knowledge_base():
    return _service.load()

def retrieve(query):
    result = _service.retrieve(query, profile="full")
    return result.path, ",".join(chunk.id for chunk in result.chunks), result.context

def retrieve_semantic(query, doc_scope=None, mode="embedding_graph"):
    profile = "full" if mode == "embedding_graph" else "vector"
    result = _service.retrieve(query, doc_scope=doc_scope, profile=profile)
    return result.path, ",".join(chunk.id for chunk in result.chunks), result.context

def generate_answer(query, context):
    return _service.generate_answer(query, context)

def add_document_to_tree(file_path, batch_size=20):
    return _service.add_document(file_path, batch_size=batch_size)
```

Preserve the existing CLI labels and destructive reset confirmation. Update `dingtalk_server.py` imports only; do not change its endpoint, authentication, payload, raw-context behavior, or bind address in this task.

- [ ] **Step 4: Run compatibility tests and commit**

Run: `python -m pytest tests/test_compatibility.py -v`

Expected: all compatibility tests pass.

```bash
git add graphrag/service.py graphrag_assistant.py dingtalk_server.py tests/test_compatibility.py
git commit -m "refactor: preserve CLI and API compatibility"
```

### Task 9: Public evaluation fixtures and runner

**Files:**
- Delete: `scripts/two_doc_retrieval_eval.py`
- Create: `scripts/retrieval_eval.py`
- Create: `examples/eval_technical_notes.txt`
- Create: `examples/eval_financial_notes.txt`
- Create: `examples/eval_questions.json`
- Create: `tests/test_public_eval.py`

- [ ] **Step 1: Write failing evaluation tests**

```python
from pathlib import Path
from scripts.retrieval_eval import evaluate, load_fixture

def test_public_fixture_has_no_absolute_paths():
    documents, questions = load_fixture(Path("examples"))
    assert len(documents) == 2 and questions
    assert all(not Path(item["path"]).is_absolute() for item in documents)

def test_evaluation_runs_all_profiles_with_same_scope_and_budget(fake_service, tmp_path):
    report = evaluate(fake_service, Path("examples/eval_questions.json"), tmp_path)
    assert set(report["profiles"]) == {"vector", "vector_keyword", "vector_graph", "vector_topic_graph", "full"}
    for question in report["questions"]:
        scopes = {row["doc_scope"] for row in question["results"]}
        budgets = {row["top_k"] for row in question["results"]}
        assert len(scopes) == 1 and len(budgets) == 1
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_public_eval.py -v`

Expected: FAIL because the public runner and fixtures do not exist.

- [ ] **Step 3: Add lawful synthetic documents and question schema**

```json
[
  {
    "id": "TECH-01",
    "document": "eval_technical_notes.txt",
    "question": "控制器如何根据温度变化调整冷却流程？",
    "expected_evidence": ["温度传感器", "控制器", "冷却阀"],
    "top_k": 4
  },
  {
    "id": "FIN-01",
    "document": "eval_financial_notes.txt",
    "question": "本期维护费用、上期金额、同比变化和原因分别是什么？",
    "expected_evidence": ["1,250.00万元", "1,000.00万元", "25%", "设备集中检修"],
    "top_k": 4
  }
]
```

Write original synthetic prose containing these facts and no company names or copied report language.

- [ ] **Step 4: Implement generic metrics and report output**

```python
def evidence_flags(context: str, expected: list[str]) -> dict[str, bool]:
    normalized = normalize_evidence(context)
    return {point: normalize_evidence(point) in normalized for point in expected}

def reciprocal_rank(selected_texts: list[str], expected: list[str]) -> float:
    for rank, text in enumerate(selected_texts, start=1):
        if any(normalize_evidence(point) in normalize_evidence(text) for point in expected):
            return 1.0 / rank
    return 0.0
```

The CLI must accept `--questions`, repeated `--document`, `--out-json`, `--out-md`, and `--embedding-backend`. Defaults must point only to repository-relative public fixtures. `--embedding-backend model` uses the configured sentence-transformer; `--embedding-backend deterministic` uses a small token-hashing backend solely for offline tests and smoke execution, and the report must record which backend produced it.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_public_eval.py -v`

Expected: public fixture and same-pipeline tests pass.

```bash
git add scripts examples tests/test_public_eval.py
git commit -m "test: publish reproducible retrieval evaluation"
```

### Task 10: CPU benchmark, documentation, and CI

**Files:**
- Create: `scripts/cpu_benchmark.py`
- Create: `.github/workflows/tests.yml`
- Modify: `README.md`
- Modify: `docs/system_design.md`
- Modify: `docs/evaluation.md`
- Modify: `GITHUB_RELEASE_CHECKLIST.md`
- Create: `tests/test_repository_hygiene.py`

- [ ] **Step 1: Write failing repository-hygiene tests**

```python
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
```

- [ ] **Step 2: Verify the hygiene tests fail**

Run: `python -m pytest tests/test_repository_hygiene.py -v`

Expected: FAIL until old implementation and documentation are cleaned.

- [ ] **Step 3: Add non-gating benchmark and CI**

```python
# scripts/cpu_benchmark.py
def main():
    started = time.perf_counter()
    service = build_benchmark_service()
    build_seconds = service.build_public_fixture()
    query_seconds = service.time_queries()
    print(json.dumps({
        "wall_seconds": time.perf_counter() - started,
        "build_seconds": build_seconds,
        "query_seconds": query_seconds,
    }, ensure_ascii=False, indent=2))
```

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    strategy:
      matrix:
        os: [windows-latest, ubuntu-latest]
        python: ["3.10", "3.12"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: python -m pip install -r requirements-dev.txt
      - run: python -m pytest -v
      - run: python -m compileall -q graphrag graphrag_assistant.py dingtalk_server.py scripts
```

- [ ] **Step 4: Rewrite documentation to match actual behavior**

README must include:

- Online first-run model download.
- Explicit `OFFLINE_MODE=1` behavior and pre-download command.
- SQLite default and non-destructive JSON migration.
- Core/eval/dev dependency separation.
- Public evaluation and CPU benchmark commands.
- Compatibility statement for CLI and Python functions.

System design must describe package boundaries and SQLite matrix recall. Evaluation docs must replace claims based on the private script with reproducible fixture instructions and clearly label older numbers as historical, non-reproducible snapshots.

- [ ] **Step 5: Run hygiene tests and commit**

Run: `python -m pytest tests/test_repository_hygiene.py -v`

Expected: 2 passed.

```bash
git add .github scripts/cpu_benchmark.py README.md docs GITHUB_RELEASE_CHECKLIST.md tests/test_repository_hygiene.py
git commit -m "docs: align setup evaluation and CI"
```

### Task 11: Full integration verification

**Files:**
- Modify only files required by failures discovered below.

- [ ] **Step 1: Run the complete automated suite**

Run: `python -m pytest -v`

Expected: all tests pass with zero failures and no network/model downloads.

- [ ] **Step 2: Run syntax compilation**

Run: `python -m compileall -q graphrag graphrag_assistant.py dingtalk_server.py scripts tests`

Expected: exit code 0 and no output.

- [ ] **Step 3: Run the public evaluation smoke test**

Run: `python scripts/retrieval_eval.py --embedding-backend deterministic --out-json .tmp/eval.json --out-md .tmp/eval.md`

Expected: exit code 0, five profiles in both outputs, and no absolute author paths.

- [ ] **Step 4: Run compatibility and leak scans**

Run: `python -m pytest tests/test_compatibility.py tests/test_repository_hygiene.py -v`

Expected: all tests pass.

Run: `rg -n "中华第一库|Super-N|Fast-N|650\.33亿元|60亿元股票回购|C:\\Users\\HUAWEI" graphrag scripts tests README.md docs/evaluation.md`

Expected: no matches except the explicit forbidden-term tuple inside `tests/test_repository_hygiene.py`; inspect that file-only result manually.

- [ ] **Step 5: Attempt a real CPU model smoke test**

Run: `python -c "from graphrag.config import AppConfig; from graphrag.embeddings import EmbeddingBackend; c=AppConfig.load(__import__('pathlib').Path.cwd()); b=EmbeddingBackend.load(c.embedding_model,c.offline_mode); print(b.encode_many(['测试文本'],1).shape)"`

Expected when model/network is available: `(1, <embedding dimension>)`.

If unavailable, record the exact `ModelUnavailableError`; do not report the real-model smoke as passed.

- [ ] **Step 6: Inspect final repository state**

Run: `git status --short`

Expected: no generated database, evaluation output, model cache, secrets, or temporary files staged.

Run: `git diff origin/main...HEAD --check`

Expected: exit code 0.

Run: `git diff --stat origin/main...HEAD`

Expected: only files described by this plan.

### Task 12: Review, synchronize, and publish main

**Files:**
- No source changes unless review finds an Important or Critical defect.

- [ ] **Step 1: Perform code review against the design and plan**

Review:

- Compatibility API and CLI.
- Migration preservation and batch atomicity.
- Profile fairness.
- Absence of fixture-specific production logic.
- CPU-only behavior and no-network tests.
- Documentation/implementation agreement.

Fix every Critical or Important issue, then rerun Task 11 in full.

- [ ] **Step 2: Confirm remote main has not diverged**

Run: `git fetch origin`

Expected: exit code 0.

Run: `git status --short --branch`

Expected: clean working tree and a normal ahead count, not diverged.

Run: `git pull --ff-only origin main`

Expected: already up to date or a clean fast-forward.

- [ ] **Step 3: Re-run verification after synchronization**

Run: `python -m pytest -v`

Expected: all tests pass with zero failures.

Run: `python -m compileall -q graphrag graphrag_assistant.py dingtalk_server.py scripts tests`

Expected: exit code 0.

- [ ] **Step 4: Push main without force**

Run: `git push origin main`

Expected: normal fast-forward update of `origin/main`.

- [ ] **Step 5: Read back remote state**

Run: `git ls-remote origin refs/heads/main`

Expected: remote SHA equals `git rev-parse HEAD`.

Run: `git status --short --branch`

Expected: `main...origin/main` with a clean working tree and no ahead/behind count.
