# Retrieval efficiency and output-selection checks

This follow-up uses the unchanged 400-document / 30-development-query pilot from
[CRUD pilot preparation](crud_pilot.md). Source metrics remain proxies, not
exhaustive relevance judgments or official CRUD-RAG results. No generation API,
held-out evaluation, weight tuning, or application-default change is introduced.

## Engineering changes

- Warm pure-vector retrieval loads candidate records by ID rather than all corpus
  records. It reuses the existing matrix cache; it adds no new record cache.
  The cold matrix build still reads the corpus. Store writes retain their existing
  invalidation behavior; transaction rollback leaves committed results intact.
  This is not a new guarantee for concurrent external database writers.
- Pure vector skips unused keyword/concept query extraction. Keyword terms are
  normalized once per query, retaining duplicate normalized terms and their weight.
  Queries without numeric tokens skip the exact-number corpus scan.
- The comparison adapter extracts concepts/builds graph edges only for graph
  profiles, and builds topics only for topic profiles. Application ingestion is
  unchanged. `project_index_components` makes this changed indexing workload visible.

The original five profiles, scoring weights, bridge algorithm, and default dynamic
selection remain unchanged. An optional `fixed_output_k` argument is applied only
after scoring, bridge expansion, candidate ordering and reranking. Its CLI form is
`--fixed-output-k`; it must be between 1 and the existing top-k budget.

## Twelve sequential arms

Use `cluster/crud_ablation.slurm` with the same exported interpreter, cache and
dataset variables as the pilot. Requests: one GPU, 4 CPUs, 16GB RAM, 30 minutes.
It does not submit itself, retry, overwrite old job directories or cancel jobs.

| Arm | Profile | Rerank | Final selection |
|---|---|---|---|
| project-vector | vector | no | ordinary top10 |
| llamaindex-vector | vector | no | ordinary top10 |
| project-vector-rerank | vector | yes | ordinary top10 |
| llamaindex-vector-rerank | vector | yes | ordinary top10 |
| project-full-rerank | full | yes | original dynamic3-6 |
| project-full-rerank-fixed10 | full | yes | fixed10 |
| project-vector-rerank-fixed3 | vector | yes | fixed3 |
| llamaindex-vector-rerank-fixed3 | vector | yes | fixed3 |
| project-full-rerank-fixed3 | full | yes | fixed3 |
| project-keyword-rerank-fixed10 | vector_keyword | yes | fixed10 |
| project-graph-rerank-fixed10 | vector_graph | yes | fixed10 |
| project-topic-graph-rerank-fixed10 | vector_topic_graph | yes | fixed10 |

Every arm retains vector candidate cap40 and top-k/bridge budget10. The fixed
override changes only final truncation. Changing ordinary `--top-k` instead would
also change full's bridge seed budget, confounding this comparison.

The new script sets `PYTHONHASHSEED=0` before Python starts and reports it in settings.
Reason: existing bridge expansion traverses a set while updating scores, so a
different interpreter hash seed can change pre-rerank ordering. Git revisions,
database constraints and identical input files cannot control that runtime order.
This is an experiment randomization control, not a checksum/contract system and
not an algorithm change. The earlier GPU pilot did not pin this seed, so don't
attribute every old/new full-profile difference solely to engineering optimization.

## Reading diagnostics

Per-query `retrieval_stage_seconds` and summary `project_stage_p50_seconds` record
project vector-data access, query embedding, vector recall, chunk reads, query-term
processing, keyword, graph, topic, exact guard, bridges, sort, rerank, and final
selection/context assembly. These are wall-clock boundaries, not isolated CUDA
kernel timings. Disabled stages have small bookkeeping durations, not model work.
Stage medians do not sum to the median total. LlamaIndex internal stages are not
instrumented and are represented as null, not zero.

Cold/warm cache, fixed arm order, storage format and index preparation still affect
timing comparisons. A single development run is not a statistical speedup claim.
Returned token counts still use the embedding tokenizer over chunk clean text,
not an actual generated prompt or billing counter.

Local verification checks the old/new deterministic vector and full runs over all
30 questions for exact ordered chunk IDs and scores. It does not establish GPU
equivalence, model quality improvements or a new held-out result. Keep all arm
results, missing arms and failures when interpreting the next cluster run.
