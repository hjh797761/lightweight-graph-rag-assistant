"""Explicit local evidence builds. Existing outputs require unfinished --resume."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphrag.config import AppConfig
from graphrag.errors import IndexCapabilityError
from graphrag.evidence_store import EvidenceStore, inspect_index
from graphrag.ingestion import make_doc_id
from graphrag.service import GraphRAGService
from graphrag.structure import parse_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a new evidence index from source files without changing the old index.")
    parser.add_argument("--input", nargs="+", required=True, help="TXT, Markdown, DOCX or text PDF sources")
    parser.add_argument("--out", required=True, help="New SQLite output path (must not exist without --resume)")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--embedding-backend", choices=("model", "deterministic"), default="model",
                        help="deterministic is a smoke backend, not a model-quality evaluation")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--allow-download", action="store_true", help="Allow model downloads; by default use local cache only")
    parser.add_argument("--resume", action="store_true", help="Continue an unfinished evidence index with the original inputs and embedding configuration")
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    if out.suffix.lower() == ".json":
        raise ValueError("Use a SQLite output filename, such as .sqlite3; .json is reserved for legacy configuration paths")
    config = AppConfig(kb_path=out, offline_mode=not args.allow_download,
                       embedding_backend="sentence_transformer" if args.embedding_backend == "model" else "deterministic",
                       embedding_model=args.model, enable_cross_encoder=False, batch_size=args.batch_size)
    store = None
    if args.resume:
        capabilities = inspect_index(out)
        if not capabilities["evidence_ready"]:
            raise IndexCapabilityError(f"Cannot resume this index: {capabilities['reason']} Use --out with a new file.")
        store = EvidenceStore(out)
        documents = {str(d["id"]): d for d in store.list_documents()}
        pending = {id for id, document in documents.items() if document["state"] != "complete"}
        if not pending:
            raise ValueError("Index is already complete or has no prepared documents; --resume requires unfinished documents")
        store.check_embedding_configuration(config.embedding_backend, config.embedding_model)
    elif out.exists():
        raise FileExistsError(f"Output already exists: {out}; choose a new --out, or explicitly --resume an unfinished evidence index")
    inputs = [(make_doc_id(path), Path(path).resolve()) for path in args.input]
    ids = [id for id, _ in inputs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate input files")
    if args.resume and (not pending <= set(ids) or not set(ids) <= documents.keys()):
        raise ValueError("Resume input files must include every pending document and may include only original index inputs")
    parsed = [(id, path, parse_file(path, id)) for id, path in inputs]
    if args.resume:
        # Validate every supplied source, including completed documents, before writing.
        for id, path, chunks in parsed:
            if str(path) != documents[id]["path"] or path.name != documents[id]["name"]:
                raise ValueError("Document source changed; resume requires original input paths")
            store.validate_source(id, chunks)
    else:
        store = EvidenceStore.create(out)
        store.check_embedding_configuration(config.embedding_backend, config.embedding_model, record=True)
        # Prepare every full source before the first model call, retaining pending work.
        for id, path, chunks in parsed:
            store.prepare_document(id, chunks, doc_name=path.name, doc_path=str(path))
    service = GraphRAGService(config, store)
    for id, path, chunks in parsed:
        if args.resume and documents[id]["state"] == "complete":
            continue
        while True:
            processed, total = service._embed_prepared_document(id, chunks, config.batch_size)
            print(f"{path.name}: {processed}/{total}")
            if processed == total:
                break
    print(f"Evidence index complete: {out}")
    print("To switch explicitly, set these values in your application configuration (no configuration file was changed):")
    print(f"KB_PATH={out}")
    print("RETRIEVAL_PROFILE=evidence")
    print(f"EMBEDDING_BACKEND={config.embedding_backend}")
    print(f"EMBEDDING_MODEL={config.embedding_model}")
    print(f"ENABLE_CROSS_ENCODER={int(config.enable_cross_encoder)}")
    print("OFFLINE_MODE=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
