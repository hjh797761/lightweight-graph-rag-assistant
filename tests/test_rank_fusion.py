import importlib

import pytest


def fuse(rankings, k=60):
    return importlib.import_module("graphrag.rank_fusion").reciprocal_rank_fusion(rankings, k)


def test_duplicate_ids_use_dense_ranks_and_sum_channel_contributions():
    result = fuse({"vector": ["a", "a", "b"], "lexical": ["b"]})
    assert result == [
        ("b", 1 / 62 + 1 / 61, {"vector": 2, "lexical": 1}),
        ("a", 1 / 61, {"vector": 1}),
    ]


def test_ties_use_chunk_id_and_ignore_channel_insertion_order():
    first = {"vector": ["b", "a"], "lexical": ["a", "b"]}
    second = dict(reversed(list(first.items())))
    assert fuse(first) == fuse(second)
    assert [item[0] for item in fuse(first)] == ["a", "b"]


def test_empty_rankings_and_empty_channels():
    assert fuse({}) == []
    assert fuse({"vector": [], "lexical": []}) == []
    assert fuse({"vector": [], "lexical": ["x"]}, k=1) == [
        ("x", 0.5, {"lexical": 1})
    ]


@pytest.mark.parametrize("k", [True, False, 0, -1, 1.5, "60", None])
def test_invalid_k_is_rejected_even_for_empty_input(k):
    with pytest.raises(ValueError, match="positive integer"):
        fuse({}, k=k)
