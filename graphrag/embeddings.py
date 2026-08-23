import numpy as np

from .errors import ModelUnavailableError


class EmbeddingBackend:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, name: str, offline: bool, loader=None):
        if loader is None:
            from sentence_transformers import SentenceTransformer

            loader = SentenceTransformer
        try:
            return cls(loader(name, local_files_only=offline))
        except Exception as exc:
            mode = "离线缓存" if offline else "在线下载"
            raise ModelUnavailableError(f"无法通过{mode}加载模型 {name}") from exc

    def encode_many(self, texts: list[str], batch_size: int) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        values = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(values, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode_many([text], batch_size=1)[0]
