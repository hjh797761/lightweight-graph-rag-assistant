"""Prepare a development-only source-retrieval proxy, NOT official CRUD-RAG scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.comparison_metrics import validate_dataset

TASKS = {"questanswer_1doc": 1, "questanswer_2docs": 2, "questanswer_3docs": 3}
SOURCE_URL = "https://github.com/IAAR-Shanghai/CRUD_RAG/blob/main/data/crud_split/split_merged.json"
LIMITATION = (
    "Source-document retrieval proxy: news fields are generation sources, not exhaustive relevance "
    "judgments. Other retrieved documents may also be relevant; multi-source questions need not "
    "require every listed source. Reduced-corpus development pilot, not official CRUD-RAG scores "
    "or answer-quality evaluation. Exact texts are deduplicated; near duplicates are not removed."
)


def prepare(source, *, query_events=10, corpus_events=100, seed=20260918):
    if type(query_events) is not int or type(corpus_events) is not int or not 0 < query_events <= corpus_events:
        raise ValueError("require 0 < query_events <= corpus_events")
    grouped, excluded = {}, []
    for task, count in TASKS.items():
        rows = source.get(task)
        if not isinstance(rows, list):
            raise ValueError(f"missing task list: {task}")
        seen = set()
        for index, row in enumerate(rows):
            event = row.get("ID")
            reason = None
            if not isinstance(event, str) or not event.strip():
                reason = "missing_event_id"
            elif event in seen:
                raise ValueError(f"duplicate event ID in {task}: {event}")
            else:
                seen.add(event)
                for field in ["questions"] + [f"news{i}" for i in range(1, count + 1)]:
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        reason = f"missing_or_empty_{field}"
                        break
            if reason:
                excluded.append({"task": task, "row_index": index, "event_id": event, "reason": reason})
                continue
            grouped.setdefault(event, {})[task] = (index, row)
    eligible = sorted(event for event, rows in grouped.items() if len(rows) == len(TASKS))
    if len(eligible) < corpus_events:
        raise ValueError(f"only {len(eligible)} complete events; requested {corpus_events}")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen_queries = eligible[:query_events]
    chosen_corpus = eligible[:corpus_events]
    documents, queries, by_text = [], [], {}
    for event in sorted(chosen_corpus):
        for task, count in TASKS.items():
            index, row = grouped[event][task]
            labels = {}
            for i in range(1, count + 1):
                field = f"news{i}"
                text = row[field].strip()
                if text not in by_text:
                    doc = {"id": f"crud-doc-{len(documents):05d}", "text": text, "sources": []}
                    documents.append(doc)
                    by_text[text] = doc
                doc = by_text[text]
                doc["sources"].append({"event_id": event, "task": task, "row_index": index, "field": field})
                labels[doc["id"]] = 1
            if event in chosen_queries:
                queries.append({"id": f"{task}:{event}", "event_id": event, "source_task": task,
                                "source_row_index": index, "question": row["questions"].strip(),
                                "split": "dev", "relevant_documents": labels})
    metadata = {"source_url": SOURCE_URL, "relevance_kind": "source_document_proxy",
                "limitation": LIMITATION, "seed": seed, "query_event_ids": chosen_queries,
                "corpus_event_ids": chosen_corpus, "selection": "complete events, seeded shuffle; no score filtering",
                "split_policy": "development only; future test must exclude dev events and shared source texts"}
    data = {"name": "CRUD-RAG reduced source-retrieval development pilot", "purpose": "benchmark",
            "metadata": metadata, "documents": documents, "queries": queries}
    validate_dataset(data)
    audit = {"source_rows": {task: len(source[task]) for task in TASKS},
             "complete_event_count": len(eligible), "excluded_rows": excluded,
             "incomplete_event_ids": sorted(event for event, rows in grouped.items() if len(rows) != len(TASKS)),
             "query_event_ids": chosen_queries, "corpus_event_ids": chosen_corpus,
             "documents": len(documents), "queries": len(queries), "limitation": LIMITATION}
    return data, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new directory")
    parser.add_argument("--query-events", type=int, default=10)
    parser.add_argument("--corpus-events", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--source-revision", help="optional verified upstream Git revision")
    args = parser.parse_args()
    data, audit = prepare(json.loads(args.source.read_text(encoding="utf-8")),
                          query_events=args.query_events, corpus_events=args.corpus_events, seed=args.seed)
    if args.source_revision:
        data["metadata"]["source_revision"] = args.source_revision
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in (("dataset.json", data), ("audit.json", audit)):
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"documents": len(data["documents"]), "queries": len(data["queries"]),
                      "excluded_rows": len(audit["excluded_rows"]), "out": str(args.out)}))


if __name__ == "__main__":
    main()
