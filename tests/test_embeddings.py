import numpy as np

from graphrag.embeddings import EmbeddingBackend


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.asarray([[len(text), 1.0] for text in texts], dtype=np.float64)


def test_batch_encode_returns_float32_matrix():
    fake = FakeModel()
    backend = EmbeddingBackend(model=fake)

    result = backend.encode_many(["甲", "乙乙"], batch_size=2)

    assert result.dtype == np.float32
    assert result.shape == (2, 2)
    assert len(fake.calls) == 1


def test_loader_receives_offline_flag():
    seen = {}

    def loader(name, **kwargs):
        seen.update(name=name, **kwargs)
        return FakeModel()

    EmbeddingBackend.load("model-name", offline=True, loader=loader)
    assert seen == {"name": "model-name", "local_files_only": True}
