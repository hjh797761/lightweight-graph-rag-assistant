import numpy as np
import pytest

from graphrag.embeddings import DeterministicEmbeddingBackend
from graphrag.models import ChunkRecord
from graphrag.retrieval import PROFILES, Retriever, vector_recall
from graphrag.storage import KnowledgeStore


def test_profiles_are_explicit_ablations():
    assert PROFILES["vector"].use_graph is False
    assert PROFILES["vector_graph"].use_graph is True
    assert PROFILES["full"].use_bridges is True


def test_vector_recall_uses_matrix_scores():
    ids = ["a", "b"]
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    hits = vector_recall(
        np.asarray([0.9, 0.1], dtype=np.float32), ids, matrix, top_k=1
    )
    assert hits == [("a", pytest.approx(0.9))]


def test_every_profile_preserves_scope_and_budget(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.sqlite3")
    embedder = DeterministicEmbeddingBackend(dimension=32)
    texts = ["温度传感器采集数据", "控制器调节冷却阀"]
    vectors = embedder.encode_many(texts, batch_size=2)
    store.write_chunk_batch(
        "doc",
        [
            ChunkRecord(
                id=f"doc::chunk_{index:06d}",
                doc_id="doc",
                sequence=index,
                clean_text=text,
                context_text=text,
                concepts=("控制器",) if index else ("温度传感器",),
                embedding=vectors[index],
            )
            for index, text in enumerate(texts)
        ],
        next_index=2,
    )
    retriever = Retriever(store, embedder, vector_recall_k=10, final_top_k=4)

    results = [
        retriever.retrieve("如何调节？", profile=name, doc_scope="doc", top_k=4)
        for name in PROFILES
    ]

    assert all(result.requested_scope == "doc" for result in results)
    assert all(result.requested_top_k == 4 for result in results)
    assert all(len(result.chunks) <= 4 for result in results)
    assert all("[S1]" in result.context for result in results)


def test_full_profile_uses_dynamic_selection_within_shared_budget(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.sqlite3")
    embedder = DeterministicEmbeddingBackend(dimension=32)
    texts = [f"温度控制资料第{index}项" for index in range(6)]
    vectors = embedder.encode_many(texts, batch_size=6)
    store.write_chunk_batch(
        "doc",
        [
            ChunkRecord(
                id=f"doc::chunk_{index:06d}",
                doc_id="doc",
                sequence=index,
                clean_text=text,
                context_text=text,
                embedding=vectors[index],
            )
            for index, text in enumerate(texts)
        ],
        next_index=6,
    )
    retriever = Retriever(store, embedder, vector_recall_k=10, final_top_k=5)

    simple = retriever.retrieve("温度", profile="full", doc_scope="doc", top_k=5)
    complex_result = retriever.retrieve(
        "请分别比较温度变化、原因以及控制条件",
        profile="full",
        doc_scope="doc",
        top_k=5,
    )

    assert simple.requested_top_k == complex_result.requested_top_k == 5
    assert len(simple.chunks) == 3
    assert len(complex_result.chunks) == 5
