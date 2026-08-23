import sqlite3

import numpy as np
import pytest

from graphrag.models import ChunkRecord
from graphrag.storage import KnowledgeStore


def chunk(seq: int, *, chunk_id: str | None = None) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id or f"doc::chunk_{seq:06d}",
        doc_id="doc",
        sequence=seq,
        clean_text=f"text {seq}",
        context_text=f"text {seq}",
        concepts=("概念",),
        embedding=np.asarray([seq + 1.0, 1.0], dtype=np.float32),
    )


def test_batch_constraint_failure_rolls_back_chunks_and_progress(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.sqlite3")
    conflicting = [chunk(0), chunk(0, chunk_id="different-id")]

    with pytest.raises(sqlite3.IntegrityError):
        store.write_chunk_batch("doc", conflicting, next_index=2)

    assert store.count_chunks() == 0
    assert store.get_progress("doc") == 0


def test_vector_cache_is_invalidated_after_write(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.sqlite3")
    store.write_chunk_batch("doc", [chunk(0)], next_index=1)
    ids1, matrix1 = store.vector_matrix("doc")

    store.write_chunk_batch("doc", [chunk(1)], next_index=2)
    ids2, matrix2 = store.vector_matrix("doc")

    assert ids1 == ["doc::chunk_000000"]
    assert ids2 == ["doc::chunk_000000", "doc::chunk_000001"]
    assert matrix1.shape == (1, 2)
    assert matrix2.shape == (2, 2)


def test_store_round_trips_chunk_metadata(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.sqlite3")
    original = chunk(0)
    store.write_chunk_batch("doc", [original], next_index=1)

    restored = store.list_chunks("doc")

    assert len(restored) == 1
    assert restored[0].concepts == ("概念",)
    assert np.array_equal(restored[0].embedding, original.embedding)
