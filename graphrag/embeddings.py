import numpy as np
import unicodedata

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


class DeterministicEmbeddingBackend:
    """Small dependency-free backend for tests and reproducible smoke runs."""

    def __init__(self, dimension: int = 256):
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension

    def encode_many(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        del batch_size
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
            symbols = [symbol for symbol in normalized if not symbol.isspace()]
            for index, symbol in enumerate(symbols):
                bucket = (ord(symbol) + index * 17) % self.dimension
                matrix[row, bucket] += 1.0
            for index in range(len(symbols) - 1):
                bucket = (ord(symbols[index]) * 31 + ord(symbols[index + 1])) % self.dimension
                matrix[row, bucket] += 0.5
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(matrix / norms, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode_many([text], batch_size=1)[0]


class OpenAIEmbeddingBackend:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def encode_many(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        rows = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            rows.extend(item.embedding for item in response.data)
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(matrix / norms, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode_many([text], batch_size=1)[0]
