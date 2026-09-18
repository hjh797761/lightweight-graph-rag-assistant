# CRUD source-retrieval pilot implementation

Continues the approved 2026-09-17 public comparison design and the user's
2026-09-18 approval of a small public-data retrieval comparison. Execute inline
in the existing D: checkout. No remote login, automatic submission or release.

## Scope and evidence

Use CRUD-RAG's published `split_merged.json`, QA tasks with one, two and three
source news fields. These fields are generation sources, NOT exhaustive relevance
judgments. Report source-document retrieval proxy metrics, not official CRUD-RAG
scores or generation quality. Only news fields enter the corpus. Preserve task,
row and event IDs; never index questions, reference answers or thoughts.

Select 10 complete event groups (30 development queries) using seed 20260918.
Select 90 additional event groups without looking at retrieval scores; all news
from these 100 groups form one shared corpus. Deduplicate exact stripped article
texts. No test split is executed or claimed. Future held-out construction must
exclude development event IDs and shared source texts, and audit near duplicates.
This is a small, deliberately reduced corpus, not the official 80k corpus.

## Steps

- [x] Test adapter determinism, 30-query grouping, distractors, text deduplication,
  missing source audit, invalid sizes and absence of answer/thought fields.
- [x] Implement `scripts/prepare_crud_pilot.py` using standard-library JSON and
  random sampling; write a new dataset and audit file without overwrites.
- [x] Test and implement source-proxy labels and dataset metadata preservation in
  reports, plus returned context characters and BGE-tokenizer tokens. Tokens are
  diagnostic, not generator tokens or billing estimates. Measure outside timers.
- [x] Add `cluster/crud_pilot.slurm`: 1 L40, 4 CPU, 16GB, 30 minutes, five sequential
  separate-process arms: project/llama vector without rerank; both with rerank;
  project full with rerank. Require CUDA and offline cache; no retries/cancellation.
- [x] Verify CPU tests, real optional LlamaIndex adapter, Bash syntax, source-data
  validation and deterministic full-pilot dry run. Do not claim GPU pilot success.
- [ ] Package only tracked runtime files plus the private derived dataset into a
  new transfer archive. Reuse existing cluster venv and model cache, extract into
  a new folder. Do not overwrite the existing cluster project.

No dataset articles or raw experimental results are committed or pushed. Keep
all five arms and every failure in the results; do not select winners automatically.
