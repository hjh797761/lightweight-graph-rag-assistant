from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    doc_id: str
    sequence: int
    clean_text: str
    context_text: str
    chapter: str = "未分类"
    concepts: tuple[str, ...] = ()
    embedding: Any = None


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    use_keyword: bool
    use_graph: bool
    use_topic: bool
    use_exact_guard: bool
    use_bridges: bool
    dynamic_top_k: bool


@dataclass
class RetrievalResult:
    path: str
    chunks: list[ChunkRecord] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    context: str = ""
    requested_scope: str | None = None
    requested_top_k: int = 0
