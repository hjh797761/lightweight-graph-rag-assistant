from dataclasses import dataclass
from pathlib import Path
import os

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - dependency error is clearer at use site
    dotenv_values = None

from .errors import ConfigurationError


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
        return cls(
            kb_path=kb_path,
            offline_mode=_as_bool(values.get("OFFLINE_MODE")),
            embedding_backend=str(values.get("EMBEDDING_BACKEND", "sentence_transformer")),
            embedding_model=str(values.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")),
            cross_encoder_model=str(values.get("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")),
            enable_cross_encoder=_as_bool(values.get("ENABLE_CROSS_ENCODER"), True),
            vector_recall_k=int(values.get("VECTOR_RECALL_K", "40")),
            final_top_k=int(values.get("FINAL_TOP_K", "6")),
            batch_size=int(values.get("BUILD_BATCH_SIZE", "20")),
            moonshot_api_key=str(values.get("MOONSHOT_API_KEY", "")),
            moonshot_base_url=str(values.get("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")),
            moonshot_model=str(values.get("MOONSHOT_MODEL", "moonshot-v1-8k")),
        )
