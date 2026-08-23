from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

from .answering import generate_answer as call_answer_model
from .config import AppConfig
from .embeddings import EmbeddingBackend, OpenAIEmbeddingBackend
from .ingestion import ingest_file
from .migration import migrate_legacy_json
from .retrieval import Retriever
from .storage import KnowledgeStore


class GraphRAGService:
    def __init__(self, config: AppConfig, store: KnowledgeStore, *, embedder=None, retriever=None, llm_client_factory=None):
        self.config = config
        self.store = store
        self._embedder = embedder
        self._retriever = retriever
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
            else:
                self._embedder = EmbeddingBackend.load(self.config.embedding_model, self.config.offline_mode)
        return self._embedder

    def _ensure_retriever(self):
        if self._retriever is None:
            self._retriever = Retriever(self.store, self._ensure_embedder(), vector_recall_k=self.config.vector_recall_k, final_top_k=self.config.final_top_k)
        return self._retriever

    def load(self) -> dict[str, int]:
        return {str(item["id"]): int(item["processed_chunks"]) for item in self.store.list_documents()}

    def retrieve(self, query: str, doc_scope: str | None = None, profile: str = "full", top_k: int | None = None):
        return self._ensure_retriever().retrieve(query, doc_scope=doc_scope, profile=profile, top_k=top_k)

    def generate_answer(self, query: str, context: str) -> str:
        return call_answer_model(self._client(), self.config.moonshot_model, query, context)

    def add_document(self, file_path: str, batch_size: int | None = None):
        return ingest_file(file_path, self.store, self._ensure_embedder(), batch_size or self.config.batch_size)

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
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = database.with_name(f"knowledge_base.backup_{stamp}.sqlite3")
            shutil.copy2(database, backup)
        if database.exists():
            database.unlink()
        self.store = KnowledgeStore(database)
        self._retriever = None
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
    if config.kb_path.suffix.lower() == ".json" and config.kb_path.exists() and not database_path.exists():
        database_path = migrate_legacy_json(config.kb_path)
    return GraphRAGService(config, KnowledgeStore(database_path))
