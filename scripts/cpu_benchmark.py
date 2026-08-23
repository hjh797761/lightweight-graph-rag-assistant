from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphrag.config import AppConfig
from graphrag.embeddings import DeterministicEmbeddingBackend
from graphrag.ingestion import ingest_file
from graphrag.retrieval import PROFILES, Retriever
from graphrag.storage import KnowledgeStore


def main() -> None:
    config = AppConfig.load(ROOT)
    documents = [
        ROOT / "examples" / "eval_technical_notes.txt",
        ROOT / "examples" / "eval_financial_notes.txt",
    ]
    questions = json.loads(
        (ROOT / "examples" / "eval_questions.json").read_text(encoding="utf-8")
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="graphrag-benchmark-") as temporary:
        store = KnowledgeStore(Path(temporary) / "benchmark.sqlite3")
        embedder = DeterministicEmbeddingBackend()
        build_started = time.perf_counter()
        for document in documents:
            while True:
                processed, total = ingest_file(
                    document, store, embedder, config.batch_size
                )
                if processed >= total:
                    break
        build_seconds = time.perf_counter() - build_started
        retriever = Retriever(
            store,
            embedder,
            vector_recall_k=config.vector_recall_k,
            final_top_k=config.final_top_k,
        )
        query_started = time.perf_counter()
        query_count = 0
        for question in questions:
            for profile in PROFILES:
                retriever.retrieve(
                    question["question"],
                    doc_scope=question["document"],
                    profile=profile,
                    top_k=int(question.get("top_k", 4)),
                )
                query_count += 1
        query_seconds = time.perf_counter() - query_started
    print(
        json.dumps(
            {
                "backend": "deterministic",
                "cpu_only": True,
                "documents": len(documents),
                "queries": query_count,
                "wall_seconds": time.perf_counter() - started,
                "build_seconds": build_seconds,
                "query_seconds": query_seconds,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
