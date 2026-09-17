# Public comparison implementation plan

**Goal:** Build an executable whole-corpus comparison for this project and LlamaIndex, then prepare a short single-GPU cluster probe.

**Architecture:** An explicit JSON dataset supplies documents, queries, split labels and document relevance grades. Independent runner processes feed the same evaluator; optional external dependencies do not enter the normal installation. Existing application APIs remain intact. Each result contains method settings and individual rankings, including failed queries.

**Tech stack:** Python, NumPy, SQLite, optional llama-index-core, existing embedding/reranker loaders, Slurm/Bash.

The user has approved implementation and test-then-push to main. Execute inline in the existing D: checkout. The user will handle two-factor login and execute supplied commands; remote tasks are not automatically submitted.

## 1. Ranking and input tests

- [x] Add tests/test_external_comparison.py with manually computed Recall/MRR/nDCG, repeated IDs, empty results, missing queries, duplicate corpus IDs and cross-split source leakage.
- [x] Run new tests before implementation: six expected missing-module failures, with a project-local temporary directory.
- [x] Implement scripts/comparison_metrics.py: graded document relevance, validation, all-query aggregation, complete Markdown reporting.
- [x] Re-run tests. No answer-string proxy is labeled retrieval relevance.

## 2. Executable comparison

- [x] Add scripts/external_comparison.py accepting dataset path, split, system, profile, embedding backend/model, optional BGE reranker and unique output directory.
- [x] Project adapter uses full-corpus Retrieval with source doc IDs. LlamaIndex uses real VectorStoreIndex with the same pre-split texts and embedding backend.
- [x] Use separate processes for systems; keep dataset, query set, chunking, embedding, output budget and reranker visible in each report. Warm-up is recorded separately.
- [x] Add an original toy dataset examples/comparison_smoke.json explicitly labeled synthetic smoke. It validates integration, not public benchmark quality.
- [x] CPU smoke both systems using deterministic embeddings; LlamaIndex 0.14.24 returned both test query rows without errors.

## 3. Cluster preparation

- [x] Add cluster/probe_gpu.slurm with 1 GPU / 4 CPU / 32GB / 10 minutes, explicit Python path, strict shell status, offline model caches and no submission loops.
- [x] Add cluster/check_queue.sh using sinfo, sbatch --test-only, squeue --start, sacct. It never cancels jobs.
- [x] Add scripts/probe_gpu.py for CUDA allocation and optional cached BGE encode/rerank, recording duration/peak allocated memory. GPU execution is pending user submission.
- [x] Document model preparation, dataset requirements and status verification. Python AST and Bash syntax checks passed.

## 4. Verification and integration

- [x] Run full CPU tests: 37 passed, including real LlamaIndex integration. Python AST checks passed for 19 files, Bash syntax checks passed, git diff --check passed.
- [ ] Review only task files, commit and push fast-forward main after checking remote state. No Release or performance claims until real model/data results exist.
- [ ] Cluster authentication, CRUD-RAG source mapping, Qwen3-specific reranker, LightRAG LLM-based indexing and public showcase remain dependent follow-on work. Report exact readiness, not completion of the larger GPU experiment.

## Deferred comparisons

LightRAG needs a tested extraction/keyword model and original-source mapping; Microsoft GraphRAG additionally requires isolated package environment and a cost pilot. These are candidates, not implemented adapters. They cannot be marked as timed-out/defeated without a real run. Exact dataset adaptation waits for inspection of official source labels.
