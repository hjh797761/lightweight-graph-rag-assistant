"""Run one isolated full-corpus retrieval configuration, never a generation test."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphrag.embeddings import CrossEncoderReranker, DeterministicEmbeddingBackend, EmbeddingBackend
from graphrag.graph import build_topics, extract_concepts
from graphrag.ingestion import split_text
from graphrag.models import ChunkRecord
from graphrag.retrieval import PROFILES, Retriever
from graphrag.storage import KnowledgeStore
from scripts.comparison_metrics import report_markdown, summarize, validate_dataset


def _versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "sentence-transformers", "torch", "llama-index-core"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _chunks(data, embedder):
    records = []
    for position, doc in enumerate(data["documents"]):
        for sequence, (clean, context) in enumerate(split_text(doc["text"], chunk_size=800)):
            records.append(ChunkRecord(id=f"d{position}:c{sequence}", doc_id=doc["id"],
                                       sequence=sequence, clean_text=clean, context_text=context))
    # Batches bound host memory during inference; every method gets identical inputs.
    embedded = []
    for start in range(0, len(records), 32):
        batch = records[start:start + 32]
        vectors = embedder.encode_many([record.clean_text for record in batch], batch_size=32)
        if len(vectors) != len(batch):
            raise ValueError("embedding backend returned the wrong number of vectors")
        embedded.extend(replace(record, embedding=vector) for record, vector in zip(batch, vectors))
    return embedded


def _project_retriever(chunks, embedder, output, profile, candidate_k, top_k, reranker):
    store = KnowledgeStore(output / "index.sqlite3")
    by_doc = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(replace(chunk, concepts=extract_concepts(chunk.clean_text)))
    for doc_id, records in by_doc.items():
        store.write_chunk_batch(doc_id, records, len(records), doc_name=doc_id)
        store.replace_topics(doc_id, build_topics(records))
    retriever = Retriever(store, embedder, vector_recall_k=candidate_k, final_top_k=top_k, reranker=reranker)

    def retrieve(query):
        result = retriever.retrieve(query, doc_scope=None, profile=profile, top_k=top_k)
        return list(zip(result.chunks, result.scores))

    return retrieve


def run(data, output: Path, *, system="project", backend="deterministic", profile="vector",
        split="test", model="BAAI/bge-small-zh-v1.5", rerank=False, offline=True,
        top_k=10, candidate_k=40, require_cuda=False):
    validate_dataset(data)
    if system not in {"project", "llamaindex"} or profile not in PROFILES or split not in {"dev", "test"}:
        raise ValueError("unknown system, profile or split")
    if system == "llamaindex" and profile != "vector":
        raise ValueError("LlamaIndex adapter supports only the vector configuration")
    if backend not in {"model", "deterministic"}:
        raise ValueError("unknown embedding backend")
    if backend == "deterministic" and rerank:
        raise ValueError("deterministic smoke does not load a model reranker")
    if require_cuda and backend != "model":
        raise ValueError("CUDA requires a real model backend")
    if not 0 < top_k <= candidate_k:
        raise ValueError("positive top_k must be no larger than candidate_k")
    queries = [q for q in data["queries"] if q["split"] == split]
    if not queries:
        raise ValueError("requested split has no queries")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = {"system": system, "profile": profile, "split": split, "scope": "whole_corpus",
              "backend": backend, "embedding_model": model if backend == "model" else None,
              "reranker": "BAAI/bge-reranker-base" if rerank else None, "offline": offline,
              "top_k_chunk_budget": top_k, "vector_candidate_k": candidate_k,
              "chunk_characters": 800, "embedding_batch_size": 32,
              "dynamic_selection": system == "project" and profile == "full",
              "device": None}
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("MODEL_LOAD_BEGIN", flush=True)
    started = time.perf_counter()
    embedder = DeterministicEmbeddingBackend() if backend == "deterministic" else EmbeddingBackend.load(model, offline)
    sync = lambda: None
    gpu = None
    if backend == "model":
        import torch
        config["device"] = str(embedder.model.device)
        if str(embedder.model.device).startswith("cuda"):
            gpu = torch.cuda
            sync = gpu.synchronize
    else:
        config["device"] = "cpu"
    if backend == "model":
        config["embedding_max_sequence_length"] = embedder.model.max_seq_length
    if require_cuda and gpu is None:
        raise RuntimeError("CUDA required but embedding model is not on CUDA")
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    raw_reranker = CrossEncoderReranker.load("BAAI/bge-reranker-base", offline) if rerank else None
    config["reranker_device"] = str(raw_reranker.model.device) if raw_reranker else None
    if require_cuda and raw_reranker and not config["reranker_device"].startswith("cuda"):
        raise RuntimeError("CUDA required but reranker is not on CUDA")
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rerank_counts = []

    def bounded_reranker(query, chunks):
        selected = chunks[:candidate_k]
        rerank_counts.append(len(selected))
        return raw_reranker(query, selected)

    reranker = bounded_reranker if raw_reranker else None
    sync()
    load_seconds = time.perf_counter() - started
    if gpu:
        gpu.reset_peak_memory_stats()
    print("CORPUS_EMBEDDING_BEGIN", flush=True)
    started = time.perf_counter()
    chunks = _chunks(data, embedder)
    sync()
    encode_seconds = time.perf_counter() - started
    print(f"INDEX_BUILD_BEGIN chunks={len(chunks)}", flush=True)
    started = time.perf_counter()
    if system == "project":
        retrieve = _project_retriever(chunks, embedder, output, profile, candidate_k, top_k, reranker)
    else:
        from scripts.comparison_adapters import llamaindex_retriever
        retrieve, index = llamaindex_retriever(chunks, embedder, candidate_k=candidate_k, top_k=top_k, reranker=reranker)
        index.storage_context.persist(persist_dir=str(output / "index"))
    sync()
    build_seconds = time.perf_counter() - started
    index_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file() and path.name != "run_config.json")
    started = time.perf_counter()
    retrieve("初始化检索缓存")
    sync()
    warmup_seconds = time.perf_counter() - started
    print(f"QUERIES_BEGIN count={len(queries)}", flush=True)
    tokenizer = embedder.model.tokenizer if backend == "model" else None
    rows = []
    with (output / "per_query.jsonl").open("x", encoding="utf-8") as stream:
        for query in queries:
            sync()
            started = time.perf_counter()
            rerank_counts.clear()
            error, found = None, []
            try:
                found = retrieve(query["question"])
                sync()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            seconds = time.perf_counter() - started
            context_text = "\n\n".join(c.clean_text for c, _ in found)
            context_tokens = len(tokenizer.encode(context_text, add_special_tokens=False,
                                                  truncation=False, verbose=False)) if tokenizer else None
            row = {"query_id": query["id"], "document_ids": list(dict.fromkeys(c.doc_id for c, _ in found)),
                   "chunks": [{"id": c.id, "document_id": c.doc_id, "score": float(score)} for c, score in found],
                   "returned_chunks": len(found), "reranked_candidates": rerank_counts[-1] if rerank_counts else None,
                   "seconds": seconds, "error": error,
                   "returned_context_characters": len(context_text),
                   "returned_context_embedding_tokens": context_tokens}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            print(f"QUERY_DONE {len(rows)}/{len(queries)} error={error}", flush=True)
    summary = summarize(queries, rows)
    metadata = data.get("metadata", {})
    if metadata.get("relevance_kind") == "source_document_proxy":
        summary = {(f"source_{key}" if key.startswith(("recall@", "mrr@", "ndcg@")) else key): value
                   for key, value in summary.items()}
    summary.update({"model_load_seconds": load_seconds, "shared_chunk_embedding_seconds": encode_seconds,
                    "index_build_seconds_excluding_embedding": build_seconds,
                    "index_build_seconds_including_embedding": encode_seconds + build_seconds,
                    "warmup_seconds": warmup_seconds, "index_bytes": index_bytes,
                    "mean_returned_chunks": sum(row["returned_chunks"] for row in rows) / len(rows),
                    "mean_returned_context_characters": sum(r["returned_context_characters"] for r in rows) / len(rows),
                    "mean_returned_context_embedding_tokens": sum(r["returned_context_embedding_tokens"] for r in rows) / len(rows) if tokenizer else None,
                    "peak_torch_allocated_gpu_bytes": gpu.max_memory_allocated() if gpu else None})
    report = {"dataset": data["name"], "purpose": "smoke" if backend == "deterministic" else data["purpose"],
              "created_utc": datetime.now(timezone.utc).isoformat(), "settings": config,
              "environment": _versions(), "corpus_document_ids": [doc["id"] for doc in data["documents"]],
              "dataset_metadata": metadata, "queries": queries, "summary": summary, "rows": rows}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "report.md").write_text(report_markdown(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new directory; existing output is never overwritten")
    parser.add_argument("--system", choices=("project", "llamaindex"), default="project")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="vector")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--embedding-backend", choices=("deterministic", "model"), default="deterministic")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--require-cuda", action="store_true", help="fail instead of silently running on CPU")
    parser.add_argument("--allow-download", action="store_true", help="default only loads local model cache")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=40)
    args = parser.parse_args()
    report = run(json.loads(args.dataset.read_text(encoding="utf-8")), args.out, system=args.system,
                 profile=args.profile, split=args.split, backend=args.embedding_backend, model=args.model,
                 rerank=args.rerank, offline=not args.allow_download, top_k=args.top_k,
                 candidate_k=args.candidate_k, require_cuda=args.require_cuda)
    print(f"{report['purpose']}: {args.out / 'report.md'}; failed_queries={report['summary']['failed_queries']}")
    if report["summary"]["failed_queries"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
