from collections import Counter
import re


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
