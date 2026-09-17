"""Document-ranking metrics for explicitly labelled retrieval datasets."""
from __future__ import annotations

import math
from statistics import mean

import numpy as np


def _relevance(values: dict) -> dict[str, int]:
    if not isinstance(values, dict) or not values:
        raise ValueError("relevance labels must be a nonempty mapping")
    if any(not isinstance(k, str) or not k or type(v) is not int or not 0 <= v <= 4
           for k, v in values.items()):
        raise ValueError("relevance grades must be integers from 0 to 4 with string IDs")
    positives = {key: value for key, value in values.items() if value > 0}
    if not positives:
        raise ValueError("relevance must include at least one positive document")
    return positives


def ranking_metrics(ranking: list[str], relevance: dict[str, int], k: int) -> dict[str, float]:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    positives = _relevance(relevance)
    selected = list(dict.fromkeys(ranking))[:k]
    hits = [index for index, doc_id in enumerate(selected, 1) if doc_id in positives]
    dcg = sum((2 ** positives.get(doc_id, 0) - 1) / math.log2(index + 1)
              for index, doc_id in enumerate(selected, 1))
    ideal = sum((2 ** grade - 1) / math.log2(index + 1)
                for index, grade in enumerate(sorted(positives.values(), reverse=True)[:k], 1))
    return {"recall": len(hits) / len(positives), "mrr": 1 / hits[0] if hits else 0.0,
            "ndcg": dcg / ideal}


def validate_dataset(data: dict) -> None:
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        raise ValueError("dataset name is required")
    if data.get("purpose") not in {"smoke", "benchmark"}:
        raise ValueError("purpose must be smoke or benchmark")
    docs, queries = data.get("documents"), data.get("queries")
    if not isinstance(docs, list) or not docs or not isinstance(queries, list) or not queries:
        raise ValueError("documents and queries must be nonempty lists")
    doc_ids = set()
    for doc in docs:
        if not isinstance(doc.get("id"), str) or not doc["id"]:
            raise ValueError("document ID must be a nonempty string")
        if doc["id"] in doc_ids:
            raise ValueError("duplicate document ID")
        if not isinstance(doc.get("text"), str) or not doc["text"].strip():
            raise ValueError("document text must be nonempty")
        doc_ids.add(doc["id"])
    query_ids, doc_splits = set(), {}
    for query in queries:
        query_id = query.get("id")
        if not isinstance(query_id, str) or not query_id or query_id in query_ids:
            raise ValueError("query IDs must be nonempty and unique")
        query_ids.add(query_id)
        if not isinstance(query.get("question"), str) or not query["question"].strip():
            raise ValueError("query question must be nonempty")
        split = query.get("split")
        if split not in {"dev", "test"}:
            raise ValueError("query split must be dev or test")
        relevance = query.get("relevant_documents")
        positives = _relevance(relevance)
        if set(relevance) - doc_ids:
            raise ValueError("relevance references unknown document IDs")
        for doc_id in positives:
            if doc_id in doc_splits and doc_splits[doc_id] != split:
                raise ValueError("source document crosses dev/test split")
            doc_splits[doc_id] = split


def summarize(queries: list[dict], rows: list[dict]) -> dict:
    expected = [q["id"] for q in queries]
    actual = [r["query_id"] for r in rows]
    if not expected or len(set(expected)) != len(expected) or len(set(actual)) != len(actual) or set(expected) != set(actual):
        raise ValueError("query results must match the complete requested query set exactly")
    by_id = {row["query_id"]: row for row in rows}
    metrics = {f"{metric}@{k}": [] for k in (5, 10) for metric in ("recall", "mrr", "ndcg")}
    for query in queries:
        row = by_id[query["id"]]
        ranking = [] if row.get("error") else row["document_ids"]
        for k in (5, 10):
            for metric, value in ranking_metrics(ranking, query["relevant_documents"], k).items():
                metrics[f"{metric}@{k}"].append(value)
    seconds = [float(row["seconds"]) for row in rows]
    if any(not math.isfinite(value) or value < 0 for value in seconds):
        raise ValueError("query duration must be finite and nonnegative")
    successful_seconds = [float(row["seconds"]) for row in rows if not row.get("error")]
    return {
        **{key: mean(values) for key, values in metrics.items()},
        "queries": len(queries), "failed_queries": sum(bool(row.get("error")) for row in rows),
        "p50_seconds": float(np.percentile(seconds, 50)),
        "p95_seconds": float(np.percentile(seconds, 95)),
        "successful_p50_seconds": float(np.percentile(successful_seconds, 50)) if successful_seconds else None,
        "successful_p95_seconds": float(np.percentile(successful_seconds, 95)) if successful_seconds else None,
        "serial_successful_queries_per_second": len(successful_seconds) / sum(seconds) if sum(seconds) > 0 else None,
    }


def report_markdown(report: dict) -> str:
    lines = ["# Retrieval comparison run", "", f"Purpose: **{report['purpose']}**",
             "", "Smoke/deterministic runs do not establish model quality or speed superiority.",
             "", "Document ranks are first occurrences in the returned chunk list. Short rankings are not padded.",
             "All queries, including failures scored as zero, enter quality metrics.", "", "## Settings", "",
             "| Setting | Value |", "|---|---|"]
    for key, value in report["settings"].items():
        lines.append(f"| {key} | {str(value).replace('|', '/').replace(chr(10), ' ')} |")
    lines += ["", "## All metrics", "", "| Metric | Value |", "|---|---:|"]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {value if value is not None else 'unavailable'} |")
    lines += ["", "Full configuration, source IDs, per-query rankings, scores and errors: `report.json`.", ""]
    return "\n".join(lines)
