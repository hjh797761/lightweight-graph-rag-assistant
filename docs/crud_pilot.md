# CRUD-RAG source-retrieval development pilot

This is a **reduced-corpus source-document proxy**, not the official CRUD-RAG
evaluation, exhaustive relevance annotation, or answer-quality evaluation.
It needs no generation API. Source:
[official CRUD-RAG repository](https://github.com/IAAR-Shanghai/CRUD_RAG),
`data/crud_split/split_merged.json`.

## Data preparation

Download the source on a connected computer, then run:

```bash
python scripts/prepare_crud_pilot.py --source /path/to/split_merged.json --out .tmp/crud-pilot
```

The output directory must be new. Defaults select 10 complete event groups for
30 development questions, and 100 event groups for the shared corpus. Selection
uses seed 20260918, never model scores. The checked 2026-09-18 input has 800/797/797
QA rows, 794 complete groups and 6 incomplete groups. This selection yields
400 distinct stripped news texts, or 537 chunks with the existing splitter.
`audit.json` records omissions, source counts and selected event IDs.

Only `news1`/`news2`/`news3` text enters the corpus. Questions, reference answers,
event summaries and generation thoughts are not indexed. Document provenance
includes original task, row index, event ID and news field. Identical stripped
texts share one ordinary document ID. Near duplicates remain; this is a limitation.
Single-, two- and three-source QA each contribute ten questions. Three sources
do not guarantee a question needs all three sources.

These news fields are generation sources, not complete judged relevance sets.
Reports therefore use `source_recall@5/10`, `source_mrr@5/10`, and
`source_ndcg@5/10`. A different retrieved article can be useful even if it is
not credited. Never compare these scores directly with official CRUD-RAG scores.

All questions are development-only. Do not use them as held-out evidence after
tuning. Later test construction must exclude these query event IDs and shared
source texts, audit near duplicates, and select without inspecting model scores.
The 90 additional events provide distractors but are not an official corpus sample
representative of the upstream 80,000-plus document retrieval setting.

## Five configurations

1. Project vector, no reranking.
2. LlamaIndex vector, no reranking.
3. Project vector, BGE reranking.
4. LlamaIndex vector, BGE reranking.
5. Project full, BGE reranking.

All use BGE-small-zh-v1.5, the same corpus/questions, 800-character splitting,
embedding batch 32, vector candidate cap 40 and output chunk budget 10. Full's
existing dynamic selection may return fewer chunks; the output limit is not a
guarantee of equal returned context. Its pre-rerank candidate list is truncated
to 40 as in the existing adapter, which is an evaluation-specific restriction.
LlamaIndex here means this specific vector configuration, not its best possible
configuration or all GraphRAG capabilities.

The Slurm script runs separate processes sequentially on one GPU with a
30-minute ceiling. It does not submit jobs itself, cancel jobs, modify environments,
download models, or retry failures. It stops on the first failing arm; existing
per-query rows remain available and skipped arms must not be called completed.
New result directories include the Slurm job ID and are never overwritten.

Export `PYTHON_BIN` (existing venv), `HF_HOME` (existing offline cache), and
`PILOT_DATASET` (absolute path to dataset.json); from this checkout run:

```bash
sbatch --test-only --partition=L40 cluster/crud_pilot.slurm
sbatch --parsable --partition=L40 cluster/crud_pilot.slurm
```

The script requests 1 GPU, 4 CPUs and 16GB RAM. Queue estimates are estimates,
not promises. Verify execution with sacct plus the generated reports. It requires
both embedding and reranking models to use CUDA instead of silently falling back
to CPU. Do not run this GPU job on the login node.

## Reading results

- All queries, including failures, enter source-proxy scores. Per-query JSONL
  records ranks, scores, returned chunks, candidate counts and failures.
- Context characters and tokens are counted after each query timer. Tokens use
  the BGE embedding tokenizer without truncation, not a generator tokenizer;
  they are a length diagnostic, not actual prompt usage or API billing.
- Query latency excludes model loading, corpus embedding, indexing and warm-up.
  Context-count/reporting overhead is excluded too. Throughput is derived serial
  retrieval throughput, not concurrent service capacity.
- Index-build time includes adapter setup/imports and persistence. Repeated cache
  state and fixed arm order can affect cold-start measurements. Do not interpret
  it as a pure algorithm comparison. Index bytes reflect different storage formats.
- GPU memory is peak PyTorch allocated bytes, not the full device footprint.
- Keep every configured arm and failure in any published comparison. Do not
  publish only the winning metric without the conditions and relevant trade-offs.

Derived data, source articles, local transfer archives and raw results stay in
ignored `.tmp/`. This repository does not redistribute the upstream news corpus.
