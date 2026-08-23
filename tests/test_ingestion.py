import numpy as np

from graphrag.ingestion import ingest_text
from graphrag.storage import KnowledgeStore


class RecordingEmbedder:
    def __init__(self):
        self.calls = 0

    def encode_many(self, texts, batch_size):
        self.calls += 1
        return np.asarray([[len(text), 1.0] for text in texts], dtype=np.float32)


def test_ingestion_encodes_and_persists_one_batch(tmp_path):
    embedder = RecordingEmbedder()
    store = KnowledgeStore(tmp_path / "kb.sqlite3")

    processed, total = ingest_text(
        "doc",
        "第一节\n温度传感器采集数据。\n\n第二节\n控制器调节冷却阀。",
        store,
        embedder,
        batch_size=20,
    )

    assert processed == total == 2
    assert embedder.calls == 1
    assert store.count_chunks() == 2
    assert store.get_progress("doc") == 2
