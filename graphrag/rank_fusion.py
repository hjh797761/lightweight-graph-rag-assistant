"""Deterministic reciprocal rank fusion of independently ordered channels."""

from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]], k: int = 60
) -> list[tuple[str, float, dict[str, int]]]:
    """Return chunk IDs, fused scores, and their observed per-channel ranks.

    Duplicate IDs within a channel count once, with dense one-based ranks.
    Missing channels contribute nothing; score ties use ascending chunk ID.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    ranks: dict[str, dict[str, int]] = {}
    # Stable channel order also makes floating-point accumulation reproducible.
    for channel in sorted(rankings):
        for rank, chunk_id in enumerate(dict.fromkeys(rankings[channel]), start=1):
            ranks.setdefault(chunk_id, {})[channel] = rank

    result = [
        (chunk_id, sum(1 / (k + rank) for rank in channel_ranks.values()), channel_ranks)
        for chunk_id, channel_ranks in ranks.items()
    ]
    return sorted(result, key=lambda item: (-item[1], item[0]))
