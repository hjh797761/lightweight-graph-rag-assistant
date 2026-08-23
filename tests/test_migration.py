from pathlib import Path

import pytest

from graphrag.errors import MigrationError
from graphrag.migration import migrate_legacy_json
from graphrag.storage import KnowledgeStore


def test_migration_preserves_json_and_counts(tmp_path: Path):
    source = tmp_path / "knowledge_base.json"
    source.write_text(
        Path("tests/fixtures/legacy_kb.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    target = migrate_legacy_json(source)

    assert source.exists()
    assert target == tmp_path / "knowledge_base.sqlite3"
    store = KnowledgeStore(target)
    assert store.count_chunks() == 2
    assert store.get_progress("legacy-doc") == 2
    assert store.load_graph()["温度传感器"]["控制器"] == 3
    assert store.list_topics("legacy-doc")[0]["title"] == "温控流程"


def test_failed_migration_does_not_replace_target(tmp_path: Path):
    source = tmp_path / "knowledge_base.json"
    source.write_text("{broken", encoding="utf-8")

    with pytest.raises(MigrationError):
        migrate_legacy_json(source)

    assert source.read_text(encoding="utf-8") == "{broken"
    assert not (tmp_path / "knowledge_base.sqlite3").exists()
