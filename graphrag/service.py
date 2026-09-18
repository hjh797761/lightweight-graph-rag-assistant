from __future__ import annotations

from datetime import datetime
from dataclasses import replace
from pathlib import Path
import shutil

from .answering import generate_answer as call_answer_model
from .config import AppConfig
from .embeddings import CrossEncoderReranker, DeterministicEmbeddingBackend, EmbeddingBackend, OpenAIEmbeddingBackend
from .errors import IndexCapabilityError
from .evidence_retrieval import EvidenceRetriever
from .evidence_store import EvidenceStore, inspect_index
from .ingestion import ingest_file, make_doc_id
from .migration import migrate_legacy_json
from .retrieval import PROFILES, Retriever
from .storage import KnowledgeStore
from .structure import parse_file


def _check_profile(capabilities, profile):
    if profile == "evidence":
        if not capabilities["evidence_ready"]:
            raise IndexCapabilityError(
                f"{capabilities['reason']} Build explicitly with python scripts/build_evidence_index.py "
                "--input SOURCE --out NEW_FILE.sqlite3; keep the original index, or select a legacy RETRIEVAL_PROFILE.")
        return
    if profile not in PROFILES:
        raise ValueError(f"Unknown retrieval profile: {profile}")
    settings = PROFILES[profile]
    if not capabilities["vector"]:
        raise IndexCapabilityError(str(capabilities["reason"]))
    if settings.use_graph and not capabilities["legacy_graph"]:
        raise IndexCapabilityError(f"Profile {profile} requires legacy graph data; choose evidence/vector or the original legacy index")
    if settings.use_topic and not capabilities["legacy_topic"]:
        raise IndexCapabilityError(f"Profile {profile} requires legacy topic data; choose evidence/vector or the original legacy index")


class GraphRAGService:
    def __init__(self, config: AppConfig, store: KnowledgeStore | EvidenceStore, *, embedder=None, retriever=None, llm_client_factory=None, count_tokens=None):
        self.config = config
        self.store = store
        self._embedder = embedder
        self._retriever = retriever
        self._retrievers = {}
        self._count_tokens = count_tokens
        self._llm_client_factory = llm_client_factory
        self._llm_client = None

    def _client(self):
        if self._llm_client is None:
            if self._llm_client_factory is not None:
                self._llm_client = self._llm_client_factory()
            else:
                if not self.config.moonshot_api_key:
                    raise RuntimeError("请先设置环境变量 MOONSHOT_API_KEY。")
                from openai import OpenAI
                self._llm_client = OpenAI(api_key=self.config.moonshot_api_key, base_url=self.config.moonshot_base_url, timeout=60)
        return self._llm_client

    def _ensure_embedder(self):
        if self._embedder is None:
            if self.config.embedding_backend == "moonshot":
                self._embedder = OpenAIEmbeddingBackend(self._client(), self.config.embedding_model)
            elif self.config.embedding_backend == "deterministic":
                self._embedder = DeterministicEmbeddingBackend()
            else:
                self._embedder = EmbeddingBackend.load(self.config.embedding_model, self.config.offline_mode)
        return self._embedder

    def _ensure_retriever(self, profile):
        family = "evidence" if profile == "evidence" else "legacy"
        provided_family = "evidence" if isinstance(self._retriever, EvidenceRetriever) else "legacy"
        if self._retriever is not None and family == provided_family:
            return self._retriever
        if family not in self._retrievers:
            reranker = None
            if self.config.enable_cross_encoder:
                reranker = CrossEncoderReranker.load(
                    self.config.cross_encoder_model, self.config.offline_mode
                )
            if family == "evidence":
                self._retrievers[family] = EvidenceRetriever(
                    self.store, self._ensure_embedder(), reranker=reranker,
                    options=self.config.evidence_options, count_tokens=self._count_tokens)
            else:
                self._retrievers[family] = Retriever(
                    self.store, self._ensure_embedder(), vector_recall_k=self.config.vector_recall_k,
                    final_top_k=self.config.final_top_k, reranker=reranker)
        return self._retrievers[family]

    def load(self) -> dict[str, int]:
        return {str(item["id"]): int(item["processed_chunks"]) for item in self.store.list_documents()}

    def retrieve(self, query: str, doc_scope: str | None = None, profile: str | None = None, top_k: int | None = None):
        profile = self.config.retrieval_profile if profile is None else profile
        _check_profile(inspect_index(self.store.path), profile)
        if isinstance(self.store, EvidenceStore):
            self.store.check_embedding_configuration(self.config.embedding_backend, self.config.embedding_model)
        retriever = self._ensure_retriever(profile)
        if isinstance(retriever, EvidenceRetriever):
            return retriever.retrieve(query, doc_scope=doc_scope, top_k=top_k)
        return retriever.retrieve(query, doc_scope=doc_scope, profile=profile, top_k=top_k)

    def generate_answer(self, query: str, context: str) -> str:
        return call_answer_model(self._client(), self.config.moonshot_model, query, context)

    def add_document(self, file_path: str, batch_size: int | None = None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(self.store, EvidenceStore):
            return ingest_file(file_path, self.store, self._ensure_embedder(), batch_size)
        path = Path(file_path).resolve()
        doc_id = make_doc_id(path)
        parsed = parse_file(path, doc_id)
        self.store.check_embedding_configuration(self.config.embedding_backend, self.config.embedding_model)
        self.store.prepare_document(doc_id, parsed, doc_name=path.name, doc_path=str(path))
        return self._embed_prepared_document(doc_id, parsed, batch_size)

    def _embed_prepared_document(self, doc_id, parsed, batch_size):
        """One batch from a fully parsed and validated document."""
        self.store.check_embedding_configuration(self.config.embedding_backend, self.config.embedding_model, record=True)
        start = self.store.get_progress(doc_id)
        selected = parsed[start:start + batch_size]
        if start == len(parsed) and parsed:
            return start, len(parsed)
        vectors = self._ensure_embedder().encode_many(
            [chunk.clean_text for chunk in selected], batch_size=batch_size) if selected else []
        if len(vectors) != len(selected):
            raise ValueError("Embedding count does not match parsed chunks")
        chunks = [replace(chunk, embedding=vector) for chunk, vector in zip(selected, vectors)]
        next_index = start + len(chunks)
        self.store.write_document_batch(doc_id, chunks, next_index, len(parsed))
        return next_index, len(parsed)

    def list_documents(self):
        return self.store.list_documents()

    def list_tree(self) -> dict[str, dict[str, list[str]]]:
        tree: dict[str, dict[str, list[str]]] = {}
        for document in self.store.list_documents():
            doc_id = str(document["id"])
            chapters: dict[str, list[str]] = {}
            for chunk in self.store.list_chunks(doc_id):
                concepts = chapters.setdefault(chunk.chapter, [])
                for concept in chunk.concepts:
                    if concept not in concepts:
                        concepts.append(concept)
            tree[doc_id] = chapters
        return tree

    def reset(self, make_backup: bool = True) -> Path | None:
        database = self.store.path
        backup = None
        if database.exists() and make_backup:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = database.with_name(f"{database.stem}.backup_{stamp}{database.suffix}")
            with database.open("rb") as source, backup.open("xb") as target:
                shutil.copyfileobj(source, target)
        if database.exists():
            database.unlink()
        self.store = EvidenceStore.create(database) if isinstance(self.store, EvidenceStore) else KnowledgeStore(database)
        self._retriever = None
        self._retrievers.clear()
        return backup


class LazyService:
    def __init__(self, factory):
        self._factory = factory
        self._instance = None

    def _get(self):
        if self._instance is None:
            self._instance = self._factory()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get(), name)


def build_default_service(project_root: str | Path | None = None) -> GraphRAGService:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    config = AppConfig.load(root)
    database_path = config.database_path
    capabilities = inspect_index(database_path)
    if not capabilities["exists"]:
        if config.kb_path.suffix.lower() == ".json" and config.kb_path.exists():
            if config.retrieval_profile == "evidence":
                _check_profile(capabilities, "evidence")
            database_path = migrate_legacy_json(config.kb_path)
            capabilities = inspect_index(database_path)
        else:
            store = EvidenceStore.create(database_path) if config.retrieval_profile == "evidence" else KnowledgeStore(database_path)
            return GraphRAGService(config, store)
    _check_profile(capabilities, config.retrieval_profile)
    if capabilities["evidence_ready"]:
        store = EvidenceStore(database_path)
    else:
        # Existing legacy indexes are inspected before opening; avoid constructor DDL.
        store = KnowledgeStore.__new__(KnowledgeStore)
        store.path, store._vector_cache = database_path, {}
    return GraphRAGService(config, store)
