from __future__ import annotations

from collections import defaultdict
import re
import time

import numpy as np

from .graph import extract_concepts
from .models import ChunkRecord, RetrievalProfile, RetrievalResult
from .normalization import normalize_evidence


PROFILES = {
    "vector": RetrievalProfile("vector", False, False, False, False, False, False),
    "vector_keyword": RetrievalProfile(
        "vector_keyword", True, False, False, False, False, False
    ),
    "vector_graph": RetrievalProfile(
        "vector_graph", False, True, False, False, False, False
    ),
    "vector_topic_graph": RetrievalProfile(
        "vector_topic_graph", False, True, True, False, False, False
    ),
    "full": RetrievalProfile("full", True, True, True, True, True, True),
}


def vector_recall(query_vector, ids, matrix, top_k):
    if not ids or matrix.size == 0:
        return []
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if matrix.shape[1] != query.size:
        raise ValueError(
            f"查询向量维度 {query.size} 与知识库维度 {matrix.shape[1]} 不一致"
        )
    scores = matrix @ query
    order = np.argsort(scores)[::-1][: max(0, top_k)]
    return [(ids[index], float(scores[index])) for index in order]


def _terms(text: str) -> tuple[str, ...]:
    normalized = normalize_evidence(text)
    ascii_terms = re.findall(r"[a-z0-9_-]{2,}", normalized)
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    ngrams = [chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))]
    return tuple(dict.fromkeys([*ascii_terms, *ngrams, *extract_concepts(text)]))


def _keyword_score(query_terms: tuple[str, ...], text: str, *, normalized_terms=None) -> float:
    if not query_terms:
        return 0.0
    normalized = normalize_evidence(text)
    if normalized_terms is None:
        normalized_terms = tuple(normalize_evidence(term) for term in query_terms)
    hits = sum(1 for term in normalized_terms if term in normalized)
    return hits / len(query_terms)


def _selection_limit(query: str, budget: int, dynamic: bool) -> int:
    if not dynamic:
        return budget
    markers = ("分别", "比较", "原因", "条件", "步骤", "影响", "以及", "同时")
    complexity = sum(1 for marker in markers if marker in query)
    complexity += max(0, len(re.findall(r"[？?；;]", query)) - 1)
    return min(budget, 3 + min(complexity, 3))


class Retriever:
    def __init__(
        self,
        store,
        embedder,
        *,
        vector_recall_k: int = 40,
        final_top_k: int = 6,
        reranker=None,
    ):
        self.store = store
        self.embedder = embedder
        self.vector_recall_k = vector_recall_k
        self.final_top_k = final_top_k
        self.reranker = reranker

    def _resolve_scope(self, doc_scope: str | None) -> str | None:
        if not doc_scope:
            return None
        for document in self.store.list_documents():
            if doc_scope == document["id"]:
                return str(document["id"])
            haystack = f"{document['id']} {document['name']} {document['path']}"
            if doc_scope in haystack:
                return str(document["id"])
        return doc_scope

    def retrieve(
        self,
        query: str,
        *,
        doc_scope: str | None = None,
        profile: str = "full",
        top_k: int | None = None,
        fixed_output_k: int | None = None,
        stage_times: dict[str, float] | None = None,
    ) -> RetrievalResult:
        if profile not in PROFILES:
            raise ValueError(f"未知检索配置: {profile}")
        settings = PROFILES[profile]
        budget = int(top_k or self.final_top_k)
        if fixed_output_k is not None and (type(fixed_output_k) is not int or not 0 < fixed_output_k <= budget):
            raise ValueError("fixed_output_k must be a positive integer within top_k")
        if stage_times is not None:
            stage_times.clear()
        stage_started = time.perf_counter() if stage_times is not None else 0.0

        def checkpoint(name):
            nonlocal stage_started
            if stage_times is not None:
                now = time.perf_counter()
                stage_times[name] = now - stage_started
                stage_started = now

        resolved_scope = self._resolve_scope(doc_scope)
        ids, matrix = self.store.vector_matrix(resolved_scope)
        checkpoint("vector_data")
        query_vector = self.embedder.encode_one(query)
        checkpoint("query_embedding")
        scores: dict[str, float] = defaultdict(float)
        for chunk_id, score in vector_recall(
            query_vector, ids, matrix, self.vector_recall_k
        ):
            scores[chunk_id] = 0.70 * score
        checkpoint("vector_recall")

        needs_corpus = any((settings.use_keyword, settings.use_graph, settings.use_topic,
                            settings.use_exact_guard, settings.use_bridges))
        chunks = self.store.list_chunks(resolved_scope, chunk_ids=None if needs_corpus else list(scores))
        chunk_by_id = {chunk.id: chunk for chunk in chunks}
        checkpoint("chunk_data")

        query_terms = _terms(query) if settings.use_keyword or settings.use_graph else ()
        checkpoint("query_terms")
        if settings.use_keyword:
            normalized_terms = tuple(normalize_evidence(term) for term in query_terms)
            for chunk in chunks:
                scores[chunk.id] += 0.20 * _keyword_score(query_terms, chunk.clean_text, normalized_terms=normalized_terms)
        checkpoint("keyword")

        if settings.use_graph:
            graph = self.store.load_graph()
            related = set(query_terms)
            for term in query_terms:
                for concept, neighbors in graph.items():
                    if term in concept or concept in term:
                        related.add(concept)
                        related.update(neighbors)
            for chunk in chunks:
                hits = sum(
                    1
                    for concept in chunk.concepts
                    if any(term in concept or concept in term for term in related)
                )
                if hits:
                    scores[chunk.id] += min(0.20, hits * 0.07)
        checkpoint("graph")

        if settings.use_topic:
            for topic in self.store.list_topics(resolved_scope):
                centroid = np.asarray(topic["centroid"], dtype=np.float32)
                if centroid.size != query_vector.size:
                    continue
                topic_score = float(centroid @ query_vector)
                for chunk_id in topic["chunk_ids"]:
                    if chunk_id in chunk_by_id:
                        scores[chunk_id] += max(0.0, topic_score) * 0.08
        checkpoint("topic")

        if settings.use_exact_guard:
            exact_tokens = re.findall(r"\d+(?:\.\d+)?%?", normalize_evidence(query))
            if exact_tokens:
                for chunk in chunks:
                    normalized = normalize_evidence(chunk.clean_text)
                    if all(token in normalized for token in exact_tokens):
                        scores[chunk.id] += 0.18
        checkpoint("exact_guard")

        if settings.use_bridges and scores:
            selected_ids = set(
                sorted(scores, key=scores.get, reverse=True)[: max(budget, 2)]
            )
            by_position = {(chunk.doc_id, chunk.sequence): chunk for chunk in chunks}
            for chunk_id in tuple(selected_ids):
                chunk = chunk_by_id[chunk_id]
                for offset in (-1, 1):
                    neighbor = by_position.get((chunk.doc_id, chunk.sequence + offset))
                    if neighbor:
                        scores[neighbor.id] = max(scores[neighbor.id], scores[chunk_id] * 0.82)
        checkpoint("bridges")

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        checkpoint("candidate_sort")
        if self.reranker and ranked:
            candidate_chunks = [chunk_by_id[chunk_id] for chunk_id, _ in ranked]
            reranked = self.reranker(query, candidate_chunks)
            ranked = [(chunk.id, float(score)) for chunk, score in reranked]
        checkpoint("rerank")
        limit = fixed_output_k if fixed_output_k is not None else _selection_limit(query, budget, settings.dynamic_top_k)
        ranked = ranked[:limit]
        selected = [chunk_by_id[chunk_id] for chunk_id, _ in ranked]
        context = self._build_context(selected)
        checkpoint("selection_context")
        return RetrievalResult(
            path=f"{profile}召回-top-{len(scores)} -> 选出-top-{len(selected)}",
            chunks=selected,
            scores=[score for _, score in ranked],
            context=context,
            requested_scope=doc_scope,
            requested_top_k=budget,
        )

    @staticmethod
    def _build_context(chunks: list[ChunkRecord]) -> str:
        blocks = []
        for index, chunk in enumerate(chunks, start=1):
            blocks.append(
                f"[S{index}] 文档={chunk.doc_id} 片段={chunk.id}\n{chunk.context_text}"
            )
        return "\n\n".join(blocks)
