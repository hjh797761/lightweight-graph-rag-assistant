# -*- coding: utf-8 -*-
"""Retrieval-only evaluation for the two-document validation set.

No LLM calls are made here. The goal is to inspect isolation and evidence hits
before spending answer-generation tokens.
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PDFS = [
    Path(r"C:\Users\HUAWEI\Desktop\人工智能行业白皮书.pdf"),
    Path(r"C:\Users\HUAWEI\Desktop\贵州茅台年度报告.pdf"),
]
QUESTIONS = PROJECT_DIR / "two_doc_eval_questions.json"
FROZEN_MAIN = PROJECT_DIR / "main.py"
FROZEN_KB = PROJECT_DIR / "eval_kbs" / "kb_eval_frozen_two_docs.json"
V2_MAIN = PROJECT_DIR / "main_experimental.py"
V2_KB = PROJECT_DIR / "eval_kbs" / "kb_eval_v2_two_docs.json"
OUT_JSON = PROJECT_DIR / "two_doc_retrieval_eval_results.json"
OUT_MD = PROJECT_DIR / "two_doc_retrieval_eval_report.md"


def compact(text, limit=360):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + "..."


def load_questions(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_naive_index(pdf_paths, chunk_size=500, stride=450):
    import fitz
    from rank_bm25 import BM25Okapi

    chunks = []
    metas = []
    for pdf_path in pdf_paths:
        with fitz.open(pdf_path) as doc:
            for page_index, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                if not text.strip():
                    continue
                for offset in range(0, len(text), stride):
                    chunk = text[offset: offset + chunk_size]
                    if chunk.strip():
                        chunks.append(chunk)
                        metas.append({
                            "doc_name": pdf_path.name,
                            "page": page_index,
                            "offset": offset,
                        })
    bm25 = BM25Okapi([list(chunk) for chunk in chunks])
    return chunks, metas, bm25


def naive_retrieve(question, chunks, metas, bm25, top_k=6, doc_scope=None):
    import numpy as np

    scores = bm25.get_scores(list(question))
    allowed = []
    if doc_scope:
        allowed = [
            i for i, meta in enumerate(metas)
            if doc_scope in meta["doc_name"] or meta["doc_name"] in doc_scope
        ]
    if allowed:
        idxs = sorted(allowed, key=lambda i: scores[i], reverse=True)[:top_k]
    else:
        idxs = np.argsort(scores)[::-1][:top_k].tolist()
    selected = [
        f"{metas[i]['doc_name']}#p{metas[i]['page']}#o{metas[i]['offset']}#c{i}"
        for i in idxs
    ]
    contexts = [
        f"【文档】{metas[i]['doc_name']} 【页码】{metas[i]['page']} 【offset】{metas[i]['offset']}\n{chunks[i]}"
        for i in idxs
    ]
    return selected, contexts


def load_rag_module(name, module_path, kb_path):
    old_kb = os.environ.get("KB_PATH")
    os.environ["KB_PATH"] = str(kb_path)
    try:
        spec = importlib.util.spec_from_file_location(name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.KB_PATH = str(kb_path)
        return module
    finally:
        if old_kb is None:
            os.environ.pop("KB_PATH", None)
        else:
            os.environ["KB_PATH"] = old_kb


def split_graph_context(context):
    text = str(context or "")
    for marker in ("【关系】", "銆愬叧绯汇€?"):
        if marker in text:
            text = text.split(marker, 1)[0]
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    return blocks


def graph_retrieve(module, question, doc_scope=None):
    if doc_scope and hasattr(module, "retrieve_semantic"):
        path, selected, context = module.retrieve_semantic(question, doc_scope=doc_scope)
    else:
        path, selected, context = module.retrieve(question)
    selected_items = [item for item in str(selected or "").split(",") if item]
    return path, selected_items, split_graph_context(context)


def normalize_for_match(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace(",", "")
    return text


def key_point_variants(point):
    point = str(point or "")
    variants = {point}
    compact_point = point.replace(" ", "")
    variants.add(compact_point)
    variants.add(point.replace(",", ""))
    variants.add(compact_point.replace(",", ""))

    # Common report formatting differences: amount/unit spacing and ESG grade spacing.
    variants.add(re.sub(r"(\d)(亿元|万元|万美元|万h800gpu小时|小时|%)", r"\1\2", compact_point, flags=re.IGNORECASE))
    variants.add(re.sub(r"([a-z]+)(级)", r"\1\2", compact_point, flags=re.IGNORECASE))

    if "现金红利650.33亿元" in point:
        variants.update({"现金红利650.33亿元", "650.33亿元"})
    if "60亿元股票回购" in point:
        variants.update({"60亿元股票回购", "60亿元"})
    return [item for item in variants if item]


def flags_for_context(contexts, key_points):
    joined = "\n".join(contexts)
    normalized_joined = normalize_for_match(joined)
    flags = {}
    for point in key_points:
        variants = key_point_variants(point)
        normalized_variants = [normalize_for_match(variant) for variant in variants]
        flags[point] = any(
            variant and (variant in joined or normalize_for_match(variant) in normalized_joined)
            for variant in variants
        ) or any(variant and variant in normalized_joined for variant in normalized_variants)
    return flags


def evaluate(question_path=QUESTIONS, pdf_paths=None, out_json=OUT_JSON, out_md=OUT_MD):
    questions = load_questions(question_path)
    pdf_paths = pdf_paths or PDFS
    naive_chunks, naive_metas, naive_bm25 = build_naive_index(pdf_paths)

    frozen = load_rag_module("two_doc_frozen", FROZEN_MAIN, FROZEN_KB)
    v2 = load_rag_module("two_doc_v2", V2_MAIN, V2_KB)
    frozen.load_knowledge_base()
    v2.load_knowledge_base()

    systems = [
        ("Naive-single", lambda q, scope: naive_retrieve(q, naive_chunks, naive_metas, naive_bm25, doc_scope=None)),
        ("Naive-doc-filter", lambda q, scope: naive_retrieve(q, naive_chunks, naive_metas, naive_bm25, doc_scope=scope)),
        ("Frozen-auto", lambda q, scope: graph_retrieve(frozen, q, doc_scope=None)),
        ("Frozen-scoped", lambda q, scope: graph_retrieve(frozen, q, doc_scope=scope)),
        ("V2-auto", lambda q, scope: graph_retrieve(v2, q, doc_scope=None)),
        ("V2-scoped", lambda q, scope: graph_retrieve(v2, q, doc_scope=scope)),
    ]

    results = []
    for row in questions:
        print(f"\n{row['id']} {row['question'][:48]}")
        for name, fn in systems:
            t0 = time.time()
            output = fn(row["question"], row.get("doc_scope"))
            if len(output) == 2:
                selected, contexts = output
                path = ""
            else:
                path, selected, contexts = output
            elapsed = time.time() - t0
            flags = flags_for_context(contexts, row.get("expected_key_points", []))
            hit = sum(1 for value in flags.values() if value)
            total = len(flags)
            print(f"  {name}: {hit}/{total} {elapsed:.2f}s")
            results.append({
                "id": row["id"],
                "category": row.get("category", ""),
                "doc_scope": row.get("doc_scope", ""),
                "question": row["question"],
                "system": name,
                "retrieval_path": path,
                "selected": selected,
                "flags": flags,
                "hit": hit,
                "total": total,
                "elapsed_seconds": round(elapsed, 2),
                "context_preview": [compact(ctx) for ctx in contexts[:4]],
            })

    Path(out_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(results, out_md)
    return results


def write_markdown(results, out_md=OUT_MD):
    by_system = {}
    for row in results:
        by_system.setdefault(row["system"], {"hit": 0, "total": 0, "elapsed": 0.0, "n": 0})
        by_system[row["system"]]["hit"] += row["hit"]
        by_system[row["system"]]["total"] += row["total"]
        by_system[row["system"]]["elapsed"] += row["elapsed_seconds"]
        by_system[row["system"]]["n"] += 1

    lines = [
        "# Two-Document Retrieval Evaluation",
        "",
        "This is retrieval-only. It checks whether each system retrieves the evidence needed for the answer.",
        "",
        "## Summary",
        "",
        "| System | Key Point Hits | Hit Rate | Avg Time |",
        "|---|---:|---:|---:|",
    ]
    for system, stats in by_system.items():
        rate = stats["hit"] / stats["total"] if stats["total"] else 0
        avg = stats["elapsed"] / stats["n"] if stats["n"] else 0
        lines.append(f"| {system} | {stats['hit']}/{stats['total']} | {rate:.1%} | {avg:.2f}s |")

    current_id = None
    for row in results:
        if row["id"] != current_id:
            current_id = row["id"]
            lines.extend([
                "",
                f"## {row['id']} {row['category']}",
                "",
                f"**Doc Scope:** {row['doc_scope']}",
                "",
                f"**Question:** {row['question']}",
                "",
            ])
        lines.extend([
            f"### {row['system']}",
            f"- hit: {row['hit']}/{row['total']}",
            f"- elapsed: {row['elapsed_seconds']}s",
            f"- path: `{row.get('retrieval_path') or 'N/A'}`",
            f"- selected: {', '.join(row.get('selected') or [])}",
            "",
            "**Key point flags:**",
        ])
        for point, ok in row["flags"].items():
            lines.append(f"- [{'x' if ok else ' '}] {point}")
        lines.extend(["", "**Context preview:**"])
        for preview in row.get("context_preview", []):
            lines.append(f"- {preview}")
        lines.append("")

    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_md}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(QUESTIONS))
    parser.add_argument("--pdf", action="append", default=[])
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    args = parser.parse_args()
    pdf_paths = [Path(item) for item in args.pdf] if args.pdf else None
    evaluate(Path(args.questions), pdf_paths=pdf_paths, out_json=Path(args.out_json), out_md=Path(args.out_md))


if __name__ == "__main__":
    main()
