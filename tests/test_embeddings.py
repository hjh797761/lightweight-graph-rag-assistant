import numpy as np

from graphrag.embeddings import CrossEncoderReranker, EmbeddingBackend, OpenAIEmbeddingBackend
from graphrag.models import ChunkRecord


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


def test_openai_compatible_backend_embeds_a_batch():
    class Item:
        def __init__(self, embedding):
            self.embedding = embedding

    class Embeddings:
        def __init__(self):
            self.seen = None

        def create(self, *, model, input):
            self.seen = (model, input)
            return type("Response", (), {"data": [Item([3.0, 4.0]), Item([0.0, 2.0])]})()

    client = type("Client", (), {"embeddings": Embeddings()})()
    backend = OpenAIEmbeddingBackend(client, "remote-model")

    result = backend.encode_many(["甲", "乙"], batch_size=10)

    assert client.embeddings.seen == ("remote-model", ["甲", "乙"])
    assert result.dtype == np.float32
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_cross_encoder_reranker_sorts_candidates():
    class FakeCrossEncoder:
        def predict(self, pairs, show_progress_bar=False):
            assert show_progress_bar is False
            assert pairs[0][0] == "问题"
            return np.asarray([0.1, 0.9], dtype=np.float32)

    chunks = [
        ChunkRecord("a", "doc", 0, "甲", "甲"),
        ChunkRecord("b", "doc", 1, "乙", "乙"),
    ]

    ranked = CrossEncoderReranker(FakeCrossEncoder())("问题", chunks)

    assert [chunk.id for chunk, _ in ranked] == ["b", "a"]
