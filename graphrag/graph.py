from collections import Counter
import re

import numpy as np


GENERIC_CONCEPTS = {
    "公司",
    "报告",
    "情况",
    "说明",
    "项目",
    "数据",
    "信息",
    "内容",
    "合计",
}


def valid_concept(value: str) -> bool:
    concept = (value or "").strip()
    if not 2 <= len(concept) <= 24 or concept in GENERIC_CONCEPTS:
        return False
    return not bool(re.fullmatch(r"[\d\W_]+", concept))


def extract_concepts(text: str, limit: int = 6) -> tuple[str, ...]:
    candidates = re.findall(r"[\u4e00-\u9fff]{2,12}|[A-Za-z][A-Za-z0-9_-]{1,23}", text or "")
    counts = Counter(item for item in candidates if valid_concept(item))
    ranked = sorted(counts, key=lambda item: (-counts[item], candidates.index(item)))
    return tuple(ranked[:limit])


def build_concept_edges(groups) -> dict[tuple[str, str], int]:
    edges: Counter[tuple[str, str]] = Counter()
    for group in groups:
        concepts = tuple(dict.fromkeys(item for item in group if valid_concept(item)))
        for source in concepts:
            for target in concepts:
                if source != target:
                    edges[(source, target)] += 1
    return dict(edges)


def build_topics(chunks, min_similarity: float = 0.78) -> list[dict[str, object]]:
    clusters: list[dict[str, object]] = []
    for chunk in sorted(chunks, key=lambda item: (item.doc_id, item.sequence)):
        vector = np.asarray(chunk.embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not vector.size or norm == 0:
            continue
        vector = vector / norm
        similarities = [float(cluster["centroid"] @ vector) for cluster in clusters]
        best_index = int(np.argmax(similarities)) if similarities else -1
        if best_index >= 0 and similarities[best_index] >= min_similarity:
            cluster = clusters[best_index]
            cluster["vectors"].append(vector)
            cluster["chunks"].append(chunk)
            centroid = np.mean(cluster["vectors"], axis=0)
            centroid_norm = float(np.linalg.norm(centroid)) or 1.0
            cluster["centroid"] = np.asarray(centroid / centroid_norm, dtype=np.float32)
        else:
            clusters.append({"centroid": vector, "vectors": [vector], "chunks": [chunk]})

    topics = []
    for index, cluster in enumerate(clusters):
        members = cluster["chunks"]
        title_candidates = [
            concept
            for chunk in members
            for concept in chunk.concepts
            if valid_concept(concept)
        ]
        title = " / ".join(dict.fromkeys(title_candidates))[:60] or members[0].chapter
        topics.append(
            {
                "id": f"{members[0].doc_id}::topic_{index:04d}",
                "title": title,
                "centroid": cluster["centroid"],
                "chunk_ids": [chunk.id for chunk in members],
            }
        )
    return topics
