from dataclasses import dataclass, field, fields
from pathlib import Path
import os

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - dependency error is clearer at use site
    dotenv_values = None

from .errors import ConfigurationError
from .models import EvidenceOptions
from .retrieval import PROFILES


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    kb_path: Path = Path("knowledge_base.json")
    offline_mode: bool = False
    embedding_backend: str = "sentence_transformer"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    cross_encoder_model: str = "BAAI/bge-reranker-base"
    enable_cross_encoder: bool = True
    vector_recall_k: int = 40
    final_top_k: int = 6
    batch_size: int = 20
    moonshot_api_key: str = ""
    moonshot_base_url: str = "https://api.moonshot.cn/v1"
    moonshot_model: str = "moonshot-v1-8k"
    retrieval_profile: str = "evidence"
    evidence_options: EvidenceOptions = field(default_factory=EvidenceOptions)

    def __post_init__(self):
        if self.retrieval_profile not in {"evidence", *PROFILES}:
            raise ConfigurationError(f"Unknown RETRIEVAL_PROFILE: {self.retrieval_profile}")
        if self.embedding_backend not in {"sentence_transformer", "moonshot", "deterministic"}:
            raise ConfigurationError(f"Unknown EMBEDDING_BACKEND: {self.embedding_backend}")
        for name in ("vector_recall_k", "final_top_k", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")
        if not isinstance(self.evidence_options, EvidenceOptions):
            raise ConfigurationError("evidence_options must be EvidenceOptions")

    @property
    def database_path(self) -> Path:
        if self.kb_path.suffix.lower() == ".json":
            return self.kb_path.with_suffix(".sqlite3")
        return self.kb_path

    @classmethod
    def load(cls, project_root: Path) -> "AppConfig":
        if dotenv_values is None:
            raise ConfigurationError("缺少 python-dotenv，请安装 requirements.txt")
        file_values = {
            key: value
            for key, value in dotenv_values(project_root / ".env").items()
            if value is not None
        }
        values = {**file_values, **os.environ}
        kb_value = values.get("KB_PATH")
        kb_path = Path(kb_value) if kb_value else project_root / "knowledge_base.json"
        if not kb_path.is_absolute():
            kb_path = project_root / kb_path
        try:
            defaults = EvidenceOptions()
            options = {}
            for item in fields(defaults):
                default = getattr(defaults, item.name)
                options[item.name] = type(default)(values.get("EVIDENCE_" + item.name.upper(), default))
            evidence_options = EvidenceOptions(**options)
            integer_values = {name: int(values.get(key, default)) for name, key, default in (
                ("vector_recall_k", "VECTOR_RECALL_K", 40), ("final_top_k", "FINAL_TOP_K", 6),
                ("batch_size", "BUILD_BATCH_SIZE", 20))}
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid retrieval/build configuration: {exc}") from exc
        return cls(
            kb_path=kb_path,
            offline_mode=_as_bool(values.get("OFFLINE_MODE")),
            embedding_backend=str(values.get("EMBEDDING_BACKEND", "sentence_transformer")),
            embedding_model=str(values.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")),
            cross_encoder_model=str(values.get("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")),
            enable_cross_encoder=_as_bool(values.get("ENABLE_CROSS_ENCODER"), True),
            **integer_values,
            moonshot_api_key=str(values.get("MOONSHOT_API_KEY", "")),
            moonshot_base_url=str(values.get("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")),
            moonshot_model=str(values.get("MOONSHOT_MODEL", "moonshot-v1-8k")),
            retrieval_profile=str(values.get("RETRIEVAL_PROFILE", "evidence")),
            evidence_options=evidence_options,
        )
