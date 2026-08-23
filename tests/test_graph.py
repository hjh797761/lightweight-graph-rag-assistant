import numpy as np

from graphrag.graph import build_concept_edges, build_topics
from graphrag.models import ChunkRecord


def test_concept_edges_are_generic_and_symmetric():
    graph = build_concept_edges([("甲概念", "乙概念", "甲概念")])
    assert graph[("甲概念", "乙概念")] == 1
    assert graph[("乙概念", "甲概念")] == 1


def test_topic_builder_groups_near_vectors_and_separates_far_vectors():
    def record(index, vector, concept):
        return ChunkRecord(
            id=f"doc::chunk_{index:06d}",
            doc_id="doc",
            sequence=index,
            clean_text=concept,
            context_text=concept,
            concepts=(concept,),
            embedding=np.asarray(vector, dtype=np.float32),
        )

    topics = build_topics(
        [
            record(0, [1.0, 0.0], "温度传感器"),
            record(1, [0.99, 0.01], "控制器"),
            record(2, [0.0, 1.0], "维护费用"),
        ],
        min_similarity=0.9,
    )

    assert len(topics) == 2
    assert sorted(len(topic["chunk_ids"]) for topic in topics) == [1, 2]
