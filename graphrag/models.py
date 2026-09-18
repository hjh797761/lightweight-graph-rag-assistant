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
class SourceBlock:
    text: str
    section_id: str | None = None
    title_path: tuple[str, ...] = ()
    locator: str = ""
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class EvidenceChunk(ChunkRecord):
    section_id: str | None = None
    title_path: tuple[str, ...] = ()
    locator: str = ""
    source_start: int = 0
    source_end: int = 0
    partial: bool = False
    source_text: str = ""


@dataclass(frozen=True)
class EvidenceLink:
    source_id: str
    target_id: str | None
    kind: str
    basis: str = ""
    status: str = "resolved"
    source_start: int = 0
    source_end: int = 0
    target_ids: tuple[str, ...] = ()


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
