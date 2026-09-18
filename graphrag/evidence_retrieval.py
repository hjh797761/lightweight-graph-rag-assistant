"""Bounded dense/lexical retrieval followed by one shared context assembler.

Timings are per-query wall-clock observations. Candidate counts describe the
bounded channels, their raw union, the capped fusion pool and candidates sent
to the reranker separately. Rerankers may omit candidates; those omissions are explicit
in the trace and cannot silently regain a core role through an RRF fallback.
"""

from dataclasses import replace
import math
from numbers import Real
from time import perf_counter

import numpy as np

from .evidence_context import assemble_context
from .models import EvidenceOptions, RetrievalResult
from .rank_fusion import reciprocal_rank_fusion


def _detail(chunk_id):
    return dict(chunk_id=chunk_id, vector_rank=None, vector_score=None,
                lexical_rank=None, lexical_score=None, fusion_rank=None,
                fusion_score=None, rerank_score=None, final_role=None,
                omission_reason=None)


class EvidenceRetriever:
    def __init__(self, store, embedder, *, reranker=None, options=None, count_tokens=None):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.options = options if options is not None else EvidenceOptions()
        self.count_tokens = count_tokens

    def _resolve_scope(self, doc_scope):
        if doc_scope is None:
            return None
        documents = self.store.list_documents()
        matches = [doc for doc in documents if doc['id'] == doc_scope]
        if not matches:
            matches = [doc for doc in documents if doc_scope in (doc['name'], doc['path'])]
        if not matches:
            raise ValueError(f'Unknown document scope: {doc_scope!r}')
        if len(matches) != 1:
            raise ValueError(f'Document scope is ambiguous: {doc_scope!r}; use a document ID or unique path')
        if matches[0]['state'] != 'complete':
            raise ValueError(f'Document {matches[0]["id"]!r} is not complete; no evidence is available')
        return matches[0]['id']

    def retrieve(self, query, doc_scope=None, top_k=None):
        if top_k is not None and (type(top_k) is not int or top_k <= 0):
            raise ValueError('top_k must be a positive integer')
        options = self.options if top_k is None else replace(self.options, max_chunks=top_k)
        stages = {key: 0. for key in ('scope', 'vector_data', 'query_embedding',
                                      'vector_recall', 'lexical', 'fusion', 'chunk_data',
                                      'rerank', 'selection_context')}
        counts = dict(vector_candidates=0, lexical_candidates=0, candidate_union=0,
                      fused_candidates=0, reranked_candidates=0)
        started = perf_counter()

        def checkpoint(stage):
            nonlocal started
            now = perf_counter()
            stages[stage] = now - started
            started = now

        def finish(selection, details):
            return RetrievalResult(
                path=f'evidence recall {counts["candidate_union"]} -> fusion {counts["fused_candidates"]}'
                     f' -> selected {len(selection.chunks)}',
                chunks=selection.chunks, scores=selection.scores, context=selection.context,
                requested_scope=doc_scope, requested_top_k=options.max_chunks,
                selection=selection, candidate_details=details, stage_seconds=stages, counts=counts)

        scope = self._resolve_scope(doc_scope)
        checkpoint('scope')
        if not query.strip():
            selection = assemble_context([], lambda _: [], options=options, count_tokens=self.count_tokens)
            checkpoint('selection_context')
            return finish(selection, [])
        ids, matrix = self.store.vector_matrix(scope)
        checkpoint('vector_data')
        if not ids:
            selection = assemble_context([], lambda _: [], options=options, count_tokens=self.count_tokens)
            checkpoint('selection_context')
            return finish(selection, [])
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids) or not np.isfinite(matrix).all():
            raise ValueError('Index vector matrix must have matching rows and finite vectors')
        vector = np.asarray(self.embedder.encode_one(query))
        checkpoint('query_embedding')
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError('Query vector must be a nonempty finite vector')
        if vector.size != matrix.shape[1]:
            raise ValueError('Query vector dimension does not match index vectors')
        # Embedders in this package produce normalized vectors; retain their dot
        # product scores without introducing a second normalization policy.
        with np.errstate(over='ignore', invalid='ignore'):
            scores = matrix @ vector
        if not np.isfinite(scores).all():
            raise ValueError('Dense vector scores must be finite')
        dense = sorted(zip(ids, map(float, scores)), key=lambda pair: (-pair[1], pair[0]))[:options.candidate_k]
        counts['vector_candidates'] = len(dense)
        checkpoint('vector_recall')
        lexical = self.store.lexical_search(query, doc_id=scope, limit=options.lexical_k)
        counts['lexical_candidates'] = len(lexical)
        checkpoint('lexical')
        fused = reciprocal_rank_fusion({'vector': [id for id, _ in dense],
                                         'lexical': [id for id, _ in lexical]}, k=options.rrf_k)
        vector_scores, lexical_scores = dict(dense), dict(lexical)
        details = []
        for rank, (id, score, ranks) in enumerate(fused, 1):
            detail = _detail(id)
            detail.update(vector_rank=ranks.get('vector'), vector_score=vector_scores.get(id),
                          lexical_rank=ranks.get('lexical'), lexical_score=lexical_scores.get(id),
                          fusion_rank=rank, fusion_score=score,
                          omission_reason='fusion_cap' if rank > options.candidate_k else None)
            details.append(detail)
        by_detail = {d['chunk_id']: d for d in details}
        counts['candidate_union'] = len(fused)
        fused = fused[:options.candidate_k]
        counts['fused_candidates'] = len(fused)
        checkpoint('fusion')
        records = self.store.list_chunks(doc_id=scope, chunk_ids=[id for id, _, _ in fused])
        by_id = {c.id: c for c in records}
        if any(id not in by_id for id, _, _ in fused):
            raise ValueError('Candidate records are unavailable in the complete document scope')
        ranked = [(by_id[id], score) for id, score, _ in fused]
        checkpoint('chunk_data')
        if self.reranker is not None and ranked:
            counts['reranked_candidates'] = len(ranked)
            reranked, seen = [], set()
            for record, score in self.reranker(query, [c for c, _ in ranked]):
                id = getattr(record, 'id', None)
                if id not in by_id or id in seen:
                    raise ValueError('reranker must return unique candidate IDs from its input')
                if isinstance(score, (bool, np.bool_)) or not isinstance(score, Real) or not math.isfinite(score):
                    raise ValueError('reranker scores must be finite non-boolean numbers')
                seen.add(id)
                score = float(score)
                # Never use reranker-returned source text, document or embedding.
                reranked.append((by_id[id], score))
                by_detail[id]['rerank_score'] = score
            for id in by_id.keys() - seen:
                by_detail[id]['omission_reason'] = 'reranker_omitted'
            ranked = sorted(reranked, key=lambda pair: (-pair[1], by_detail[pair[0].id]['fusion_rank'], pair[0].id))
        checkpoint('rerank')

        def load_links(source_ids):
            links = self.store.links_for(source_ids, doc_id=scope)
            targets = list(dict.fromkeys(id for link in links
                          if link.kind in ('explicit_reference', 'adjacent') and link.status == 'resolved'
                          for id in (link.target_ids or ((link.target_id,) if link.target_id else ()))))
            target_records = self.store.list_chunks(doc_id=scope, chunk_ids=targets) if targets else []
            target_by_id = {c.id: c for c in target_records}
            return [(link, [target_by_id[id] for id in (link.target_ids or ((link.target_id,) if link.target_id else ()))
                            if id in target_by_id]
                     if link.kind in ('explicit_reference', 'adjacent') and link.status == 'resolved' else [])
                    for link in links]

        selection = assemble_context(ranked, load_links, options=options, count_tokens=self.count_tokens)
        omission = {d['chunk_id']: d['status'] for d in selection.decisions
                    if 'chunk_id' in d and d.get('status') != 'selected'}
        for detail in details:
            if detail['omission_reason'] is None:
                detail['omission_reason'] = omission.get(detail['chunk_id'], 'not_selected')
        for record, role in zip(selection.chunks, selection.roles):
            if record.id not in by_detail:
                detail = _detail(record.id)
                details.append(detail)
                by_detail[record.id] = detail
            by_detail[record.id].update(final_role=role, omission_reason=None)
        checkpoint('selection_context')
        return finish(selection, details)
