"""Offline evidence-path evaluation with optional external/manual annotations.

Uses the application service, store and context assembler; no answer API and no
second selection policy. Source metrics are proxies, not official CRUD scores.
"""
import argparse
from dataclasses import asdict, fields
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphrag.config import AppConfig
from graphrag.embeddings import CrossEncoderReranker, DeterministicEmbeddingBackend, EmbeddingBackend
from graphrag.evidence_retrieval import EvidenceRetriever
from graphrag.evidence_store import EvidenceStore
from graphrag.models import EvidenceOptions
from graphrag.service import GraphRAGService
from graphrag.structure import parse_text


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _prepare(data):
    if not isinstance(data, dict):
        raise ValueError("dataset must be an object")
    _text(data.get("name"), "name")
    if data.get("purpose") not in ("smoke", "benchmark"):
        raise ValueError("purpose must be smoke or benchmark")
    for name in ("documents", "queries"):
        if not isinstance(data.get(name), list) or not data[name]:
            raise ValueError(f"{name} must be a nonempty list")
    parsed, doc_ids, query_ids, chunk_ids = [], set(), set(), set()
    for doc in data["documents"]:
        if not isinstance(doc, dict):
            raise ValueError("document must be an object")
        id = _text(doc.get("id"), "document id")
        if id in doc_ids:
            raise ValueError(f"duplicate document id: {id}")
        doc_ids.add(id)
        chunks = parse_text(_text(doc.get("text"), "document text"), id,
                            _text(doc.get("format", "markdown"), "document format"))
        chunk_ids.update(c.id for c in chunks)
        parsed.append((id, chunks))
    positives = {"dev": set(), "test": set()}
    for query in data["queries"]:
        if not isinstance(query, dict):
            raise ValueError("query must be an object")
        id = _text(query.get("id"), "query id")
        if id in query_ids:
            raise ValueError(f"duplicate query id: {id}")
        query_ids.add(id)
        _text(query.get("question"), "question")
        if query.get("split") not in positives:
            raise ValueError("query split must be dev or test")
        if query.get("doc_scope") is not None:
            scope = _text(query["doc_scope"], "doc_scope")
            # Inline names equal IDs; exact IDs take precedence over paths,
            # matching EvidenceRetriever._resolve_scope without using labels.
            if scope not in doc_ids and not any(scope == f"inline:{id}" for id in doc_ids):
                raise ValueError(f"Unknown document scope: {scope!r}")
        if "relevant_documents" in query:
            labels = query["relevant_documents"]
            if not isinstance(labels, dict):
                raise ValueError("relevant_documents must map document IDs to integer grades 0..4")
            for doc_id, grade in labels.items():
                if doc_id not in doc_ids or type(grade) is not int or not 0 <= grade <= 4:
                    raise ValueError("unknown relevant document or invalid grade (expected integer 0..4)")
                if grade > 0:
                    positives[query["split"]].add(doc_id)
        exhaustive = query.get("reference_targets_exhaustive", False)
        if type(exhaustive) is not bool:
            raise ValueError("reference_targets_exhaustive must be boolean")
        if exhaustive and "reference_target_ids" not in query:
            raise ValueError("exhaustive reference labels require reference_target_ids (may be empty)")
        if "reference_target_ids" in query:
            targets = query["reference_target_ids"]
            if (not isinstance(targets, list) or any(not isinstance(t, str) for t in targets)
                    or len(targets) != len(set(targets)) or not set(targets) <= chunk_ids):
                raise ValueError("reference_target_ids must be unique known chunk IDs")
    if positives["dev"] & positives["test"]:
        raise ValueError("positive source document IDs cannot cross dev/test splits")
    return parsed


def _code_state():
    try:
        root = Path(__file__).resolve().parents[1]
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                  text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"revision": revision, "dirty": dirty, "status": "available"}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None, "status": "unavailable"}


def _runtime(embedder, reranker):
    versions = {"python": platform.python_version()}
    for name in ("numpy", "sentence-transformers", "torch", "transformers"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    def describe(component):
        model = getattr(component, "model", None)
        device = getattr(model, "device", None)
        return {"implementation": type(component).__name__, "device": str(device) if device is not None else None}
    return {"versions": versions, "embedding": describe(embedder),
            "reranker": describe(reranker) if reranker is not None else None}


def _metrics(query, row):
    relevant = {id for id, grade in query.get("relevant_documents", {}).items() if grade > 0}
    sources = list(dict.fromkeys(row["document_ids"]))
    targets = set(query.get("reference_target_ids", []))
    selected = set(row["chunk_ids"])
    explicit = {id for id, role in zip(row["chunk_ids"], row["roles"]) if role == "supplement"
                and any(r["kind"] == "explicit_reference" for r in row["reasons"].get(id, []))}
    exhaustive = query.get("reference_targets_exhaustive", False)
    row["label_counts"] = {"source_positives": len(relevant), "reference_positives": len(targets),
                           "source_labels_provided": "relevant_documents" in query,
                           "reference_labels_provided": "reference_target_ids" in query,
                           "reference_targets_exhaustive": exhaustive,
                           "explicit_supplements_evaluated": len(explicit) if exhaustive else 0}
    row["false_reference_count"] = len(explicit - targets) if exhaustive else 0
    row["metrics"] = {
        "source_recall": len(relevant & set(sources)) / len(relevant) if relevant else None,
        "source_mrr": next((1. / (i + 1) for i, id in enumerate(sources) if id in relevant), 0.) if relevant else None,
        "reference_coverage": len(targets & selected) / len(targets) if targets else None,
        "reference_core_coverage": len(targets & {id for id, role in zip(row["chunk_ids"], row["roles"]) if role == "core"}) / len(targets) if targets else None,
        "reference_supplement_coverage": len(targets & {id for id, role in zip(row["chunk_ids"], row["roles"]) if role == "supplement"}) / len(targets) if targets else None,
        "false_reference_expansion": len(explicit - targets) / len(explicit) if exhaustive and explicit else None,
    }


def _aggregate(rows):
    summary = {"query_count": len(rows), "failed_queries": sum(row["error"] is not None for row in rows),
               "source_labelled_queries": sum(row["label_counts"]["source_labels_provided"] for row in rows),
               "reference_labelled_queries": sum(row["label_counts"]["reference_labels_provided"] for row in rows),
               "exhaustive_reference_queries": sum(row["label_counts"]["reference_targets_exhaustive"] for row in rows)}
    for name in ("source_recall", "source_mrr", "reference_coverage", "reference_core_coverage", "reference_supplement_coverage"):
        values = [r["metrics"][name] for r in rows if r["metrics"][name] is not None]
        summary[name] = {"value": sum(values) / len(values) if values else None,
                         "denominator": len(values), "status": "available" if values else "unavailable"}
    denominator = sum(r["label_counts"]["explicit_supplements_evaluated"] for r in rows)
    summary["false_reference_expansion"] = {
        "value": sum(r["false_reference_count"] for r in rows) / denominator if denominator else None,
        "denominator": denominator, "status": "available" if denominator else "unavailable"}
    return summary


def _markdown(report):
    # Full JSON traces keep the human and machine reports equally auditable.
    overview = {k: v for k, v in report.items() if k != "queries"}
    lines = ["# Evidence retrieval evaluation", "", report["interpretation"], "",
             "```json", json.dumps(overview, ensure_ascii=False, indent=2, allow_nan=False), "```"]
    for row in report["queries"]:
        lines.extend(["", f"## Query {row['id']}", "", "```json",
                      json.dumps({k: v for k, v in row.items() if k != "context"}, ensure_ascii=False, indent=2, allow_nan=False),
                      "```", "", "Rendered context:", "", row["context"] or "(empty)"])
    return "\n".join(lines) + "\n"


def run(dataset, out, *, embedding_backend="model", model="BAAI/bge-small-zh-v1.5",
        rerank=False, reranker_model="BAAI/bge-reranker-base", allow_download=False,
        split="dev", options=None, embedder=None, reranker=None, count_tokens=None):
    """Evaluate inline documents; injected components are recorded as such.

    For token budgets, count_tokens must be supplied by the generation tokenizer.
    The caller is responsible for the identity of injected model components.
    """
    parsed = _prepare(dataset)
    if embedding_backend not in ("deterministic", "model") or split not in ("dev", "test"):
        raise ValueError("invalid embedding backend or split")
    options = options if options is not None else EvidenceOptions()
    if not isinstance(options, EvidenceOptions):
        raise ValueError("options must be EvidenceOptions")
    if options.budget_unit == "tokens" and not callable(count_tokens):
        raise ValueError("token budgets require generator count_tokens")
    if reranker is not None and not rerank:
        raise ValueError("injected reranker requires rerank=True")
    queries = [q for q in dataset["queries"] if q["split"] == split]
    if not queries:
        raise ValueError(f"no queries in split {split}")
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}; choose a new --out")
    backend = "sentence_transformer" if embedding_backend == "model" else "deterministic"
    config = AppConfig(kb_path=out / "evidence.sqlite3", offline_mode=not allow_download,
                       embedding_backend=backend, embedding_model=model,
                       enable_cross_encoder=rerank, cross_encoder_model=reranker_model, evidence_options=options)
    code = _code_state()
    injected_embedding, injected_reranker = embedder is not None, reranker is not None
    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    store = EvidenceStore.create(config.kb_path)
    store.check_embedding_configuration(backend, model, record=True)
    for id, chunks in parsed:
        store.prepare_document(id, chunks, doc_name=id, doc_path=f"inline:{id}")
    if embedder is None:
        embedder = (DeterministicEmbeddingBackend() if embedding_backend == "deterministic"
                    else EmbeddingBackend.load(model, config.offline_mode))
    service = GraphRAGService(config, store, embedder=embedder, count_tokens=count_tokens)
    for id, chunks in parsed:
        while True:
            processed, total = service._embed_prepared_document(id, chunks, config.batch_size)
            if processed == total:
                break
    if rerank and reranker is None:
        reranker = CrossEncoderReranker.load(reranker_model, config.offline_mode)
    service = GraphRAGService(config, store, embedder=embedder, count_tokens=count_tokens,
        retriever=EvidenceRetriever(store, embedder, reranker=reranker, options=options, count_tokens=count_tokens))
    build_seconds = perf_counter() - started
    rows = []
    for query in queries:
        row = {"id": query["id"], "question": query["question"], "split": split,
               "annotations": {key: query[key] for key in ("relevant_documents", "reference_target_ids", "reference_targets_exhaustive") if key in query},
               "doc_scope": query.get("doc_scope"), "error": None,
               "chunk_ids": [], "document_ids": [], "roles": [], "scores": [],
               "core_ids": [], "supplement_ids": [], "context": "", "budget_used": 0,
               "budget_unit": options.budget_unit, "selection_status": "error", "reasons": {},
               "decisions": [], "counts": {}, "candidate_details": [], "stage_seconds": {}}
        started = perf_counter()
        try:
            result = service.retrieve(query["question"], doc_scope=query.get("doc_scope"))
            selection = result.selection
            row.update(chunk_ids=[c.id for c in result.chunks], document_ids=[c.doc_id for c in result.chunks],
                       roles=selection.roles, scores=result.scores, context=result.context,
                       core_ids=[c.id for c, role in zip(result.chunks, selection.roles) if role == "core"],
                       supplement_ids=[c.id for c, role in zip(result.chunks, selection.roles) if role == "supplement"],
                       budget_used=selection.budget_used, selection_status=selection.status,
                       reasons=selection.reasons, decisions=selection.decisions, counts=result.counts,
                       candidate_details=result.candidate_details, stage_seconds=result.stage_seconds)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_seconds"] = perf_counter() - started
        _metrics(query, row)
        rows.append(row)
    report = {"dataset": dataset["name"], "declared_purpose": dataset["purpose"],
              "purpose": "smoke" if embedding_backend == "deterministic" else dataset["purpose"],
              "code": code, "runtime": _runtime(embedder, reranker),
              "configuration": {"embedding_backend": backend, "embedding_model": model,
                  "embedding_injected": injected_embedding, "rerank": "injected" if injected_reranker else "model" if rerank else "disabled",
                  "reranker_model": reranker_model if rerank else None, "offline": config.offline_mode,
                  "split": split, "options": asdict(options), "batch_size": config.batch_size},
              "indexed_documents": len(parsed), "indexed_chunks": sum(len(c) for _, c in parsed),
              "build_seconds": build_seconds, "summary": _aggregate(rows), "queries": rows,
              "interpretation": "Source-prefixed metrics are source retrieval proxies, not exhaustive official CRUD metrics or answer quality. "
                  "Reference labels are optional external/manual annotations, never generated from links. "
                  "Coverage averages queries with positive labels, including failed queries as zero; unlabelled queries remain unavailable. "
                  "False reference expansion counts only explicit-reference supplements on exhaustive annotations; adjacency-only supplements are excluded. "
                  "Its denominator is evaluated supplements, not queries; failed retrieval has no observed expansion. "
                  "Split checks cover labelled source IDs, not near-duplicate content. Stage timings are wall time; failed queries have unavailable stage traces."}
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    markdown = _markdown(report)
    (out / "report.json").write_text(payload, encoding="utf-8")
    (out / "report.md").write_text(markdown, encoding="utf-8")
    return report


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--embedding-backend", choices=("deterministic", "model"), default="model")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    defaults = EvidenceOptions()
    for field in fields(defaults):
        if field.name != "budget_unit":
            default = getattr(defaults, field.name)
            parser.add_argument("--" + field.name.replace("_", "-"), type=type(default), default=default)
    args = vars(parser.parse_args(argv))
    options = EvidenceOptions(**{f.name: args.pop(f.name) for f in fields(defaults) if f.name != "budget_unit"})
    path = Path(args.pop("dataset"))
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    report = run(data, options=options, **args)
    print(f"Evidence report: {Path(args['out']).resolve()} ({report['purpose']}); failed queries: {report['summary']['failed_queries']}")
    return 1 if report["summary"]["failed_queries"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
