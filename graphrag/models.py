from dataclasses import dataclass, field
import math
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
    heading_end: int | None = None


@dataclass(frozen=True)
class EvidenceChunk(ChunkRecord):
    section_id: str | None = None
    title_path: tuple[str, ...] = ()
    locator: str = ""
    source_start: int = 0
    source_end: int = 0
    partial: bool = False
    source_text: str = ""
    heading_end: int | None = None
    # Original omitted whitespace since the preceding chunk in the same source block.
    source_gap_before: str = ""


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
class EvidenceOptions:
    candidate_k: int = 40
    lexical_k: int = 40
    rrf_k: int = 60
    max_chunks: int = 6
    max_supplements: int = 2
    context_budget: int = 6000
    supplement_fraction: float = 0.30
    budget_unit: str = "characters"

    def __post_init__(self):
        for name in ("candidate_k", "lexical_k", "rrf_k", "max_chunks", "context_budget"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_supplements) is not int or self.max_supplements < 0:
            raise ValueError("max_supplements must be a nonnegative integer")
        fraction = self.supplement_fraction
        if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                or not math.isfinite(fraction) or not 0 <= fraction < 1):
            raise ValueError("supplement_fraction must be finite and in [0, 1)")
        if self.budget_unit not in ("characters", "tokens"):
            raise ValueError("budget_unit must be characters or tokens")


@dataclass
class EvidenceSelection:
    chunks: list[EvidenceChunk] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    scores: list[float | None] = field(default_factory=list)
    context: str = ""
    decisions: list[dict[str, Any]] = field(default_factory=list)
    budget_used: int = 0
    budget_unit: str = "characters"
    status: str = "no_candidates"
    reasons: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def unit(self) -> str:
        return self.budget_unit


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
    scores: list[float | None] = field(default_factory=list)
    context: str = ""
    requested_scope: str | None = None
    requested_top_k: int = 0
    selection: EvidenceSelection | None = None
    candidate_details: list[dict[str, Any]] = field(default_factory=list)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
