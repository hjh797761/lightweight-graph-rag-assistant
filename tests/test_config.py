from pathlib import Path

from graphrag.config import AppConfig


def test_environment_overrides_dotenv(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text(
        "OFFLINE_MODE=1\nVECTOR_RECALL_K=7\n", encoding="utf-8"
    )
    monkeypatch.setenv("VECTOR_RECALL_K", "11")

    config = AppConfig.load(project_root=tmp_path)

    assert config.offline_mode is True
    assert config.vector_recall_k == 11


def test_json_kb_path_derives_sqlite_path(tmp_path: Path):
    config = AppConfig(kb_path=tmp_path / "knowledge_base.json")
    assert config.database_path == tmp_path / "knowledge_base.sqlite3"
