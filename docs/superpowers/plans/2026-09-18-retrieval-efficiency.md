# Retrieval efficiency and controlled selection implementation

Approved by the user after the 30-query development pilot: reduce repeated data
loading without changing ranking, prepare only profile-required index components,
then compare fixed and dynamic output selection. Work inline in the existing D:
checkout; user handles cluster authentication/submission. Do not tune weights on
the development labels or change application defaults. Test then push main.

## Design

Avoid introducing another mutable cache: the existing vector matrix supplies
vector candidate IDs; pure vector retrieval loads only those records through the
existing `list_chunks(chunk_ids=...)` API. Other profiles keep whole-corpus access.
Existing store writes invalidate the matrix, and rollback leaves it valid. This
does not introduce cross-process cache-coherence promises.

The comparison adapter only extracts/stores concepts for graph profiles and only
builds topics for topic profiles. The app's normal ingestion is unchanged. Report
the indexed components so changed build time is not misattributed.

Add optional final `fixed_output_k`, applied AFTER scoring, bridging and reranking.
The existing top_k candidate/bridge budget remains 10 in all pilot arms, so this
override isolates final truncation. Omitted override retains all existing behavior.
Add optional per-query stage durations; no GPU-kernel timing claims. Label stages
as project-only diagnostics, not directly comparable LlamaIndex internals.

## Tasks

- [x] Add failing tests for warmed vector reads restricted to candidates; no terms
  computation on vector; stable score/tie order; writes/rollback; selective index
  components; fixed selection leaving preselection candidates unchanged; reporting.
- [x] Implement minimal retrieval and adapter changes; keep application ingestion,
  weights, graph scoring and default output selection unchanged.
- [x] Compare new deterministic whole-pilot output with pre-change saved reports:
  ordered chunk IDs and scores for vector/full; run all CPU tests and Bash checks.
- [x] Prepare 12 sequential arms: original five, full fixed10, vector+rerank fixed3,
  Llama+rerank fixed3, full fixed3, keyword/graph/topic-graph+rerank fixed10.
  Use the same 400 docs / 30 dev queries. No held-out evaluation or new API calls.
- [ ] Independent review, fix concrete findings, commit/push, prepare a separate
  transfer archive. Preserve previous cluster project and results; no remote jobs
  are submitted by the assistant. No speed or quality claims without new GPU logs.

Review identified cross-process set iteration in the existing bridge algorithm.
The new Slurm script pins a common PYTHONHASHSEED and the runner records it; the
algorithm remains unchanged. A Bash-loop test checks all 12 arm arguments and seed
propagation using local Slurm stubs, without running/submitting a cluster job.
