from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphrag.config import AppConfig
from graphrag.embeddings import DeterministicEmbeddingBackend, EmbeddingBackend
from graphrag.ingestion import ingest_file
from graphrag.normalization import normalize_evidence
from graphrag.retrieval import PROFILES, Retriever
from graphrag.service import GraphRAGService
from graphrag.storage import KnowledgeStore


def load_fixture(directory: Path) -> tuple[list[dict[str, str]], list[dict]]:
    questions_path = directory / "eval_questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    document_names = list(dict.fromkeys(item["document"] for item in questions))
    documents = [
        {"id": Path(name).stem, "path": name}
        for name in document_names
    ]
    return documents, questions


def evidence_flags(context: str, expected: list[str]) -> dict[str, bool]:
    normalized = normalize_evidence(context)
    return {point: normalize_evidence(point) in normalized for point in expected}


def reciprocal_rank(selected_texts: list[str], expected: list[str]) -> float:
    for rank, text in enumerate(selected_texts, start=1):
        normalized = normalize_evidence(text)
        if any(normalize_evidence(point) in normalized for point in expected):
            return 1.0 / rank
    return 0.0


def evaluate(service, questions_path: Path, out_json: Path | None = None, *, backend: str = "injected") -> dict:
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    report = {"embedding_backend": backend, "profiles": list(PROFILES), "questions": []}
    for question in questions:
        top_k = int(question.get("top_k", 4))
        scope = question["document"]
        rows = []
        for profile in PROFILES:
            started = time.perf_counter()
            result = service.retrieve(
                question["question"],
                doc_scope=scope,
                profile=profile,
                top_k=top_k,
            )
            retrieval_seconds = time.perf_counter() - started
            selected_texts = [chunk.clean_text for chunk in result.chunks]
            flags = evidence_flags(result.context, question["expected_evidence"])
            rows.append(
                {
                    "profile": profile,
                    "path": result.path,
                    "doc_scope": result.requested_scope,
                    "top_k": result.requested_top_k,
                    "selected_chunks": [getattr(chunk, "id", "") for chunk in result.chunks],
                    "scores": [float(score) for score in getattr(result, "scores", [])],
                    "retrieval_seconds": retrieval_seconds,
                    "evidence": flags,
                    "evidence_recall": sum(flags.values()) / max(len(flags), 1),
                    "reciprocal_rank": reciprocal_rank(selected_texts, question["expected_evidence"]),
                }
            )
        report["questions"].append(
            {"id": question["id"], "question": question["question"], "results": rows}
        )
    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Public retrieval evaluation",
        "",
        f"Embedding backend: `{report['embedding_backend']}`",
        "",
        "| Question | Profile | Evidence recall | Reciprocal rank |",
        "|---|---|---:|---:|",
    ]
    for question in report["questions"]:
        for row in question["results"]:
            lines.append(
                f"| {question['id']} | {row['profile']} | {row['evidence_recall']:.3f} | {row['reciprocal_rank']:.3f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_service(root: Path, backend: str, documents: list[Path], database: Path):
    config = AppConfig.load(root)
    if backend == "deterministic":
        embedder = DeterministicEmbeddingBackend()
    else:
        embedder = EmbeddingBackend.load(config.embedding_model, config.offline_mode)
    store = KnowledgeStore(database)
    for document in documents:
        while True:
            processed, total = ingest_file(document, store, embedder, config.batch_size)
            if processed >= total:
                break
    retriever = Retriever(store, embedder, vector_recall_k=config.vector_recall_k, final_top_k=config.final_top_k)
    return GraphRAGService(replace(config, kb_path=database), store, embedder=embedder, retriever=retriever)


def main() -> None:
    root = ROOT
    parser = argparse.ArgumentParser(description="Run reproducible retrieval-profile evaluation")
    parser.add_argument("--questions", type=Path, default=root / "examples" / "eval_questions.json")
    parser.add_argument("--document", type=Path, action="append")
    parser.add_argument("--embedding-backend", choices=("model", "deterministic"), default="model")
    parser.add_argument("--out-json", type=Path, default=root / ".tmp" / "eval.json")
    parser.add_argument("--out-md", type=Path, default=root / ".tmp" / "eval.md")
    args = parser.parse_args()
    if args.document:
        documents = args.document
    else:
        fixture_documents, _ = load_fixture(args.questions.parent)
        documents = [args.questions.parent / item["path"] for item in fixture_documents]
    with tempfile.TemporaryDirectory(prefix="graphrag-eval-") as temporary:
        service = _build_service(root, args.embedding_backend, documents, Path(temporary) / "eval.sqlite3")
        report = evaluate(service, args.questions, args.out_json, backend=args.embedding_backend)
    write_markdown(report, args.out_md)
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
