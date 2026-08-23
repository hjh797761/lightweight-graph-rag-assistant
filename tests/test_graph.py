from graphrag.graph import build_concept_edges


def test_concept_edges_are_generic_and_symmetric():
    graph = build_concept_edges([("甲概念", "乙概念", "甲概念")])
    assert graph[("甲概念", "乙概念")] == 1
    assert graph[("乙概念", "甲概念")] == 1
