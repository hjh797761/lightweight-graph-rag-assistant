from pathlib import Path
import json
import os
import uuid

from .errors import MigrationError
from .storage import KnowledgeStore


def migrate_legacy_json(source: str | Path) -> Path:
    source = Path(source)
    target = source.with_suffix(".sqlite3")
    if target.exists():
        return target
    temporary = target.with_name(f"{target.name}.migrating-{uuid.uuid4().hex}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("旧知识库根节点必须是对象")
        store = KnowledgeStore(temporary)
        store.import_legacy(data)
        expected = len(data.get("chunks", {}))
        actual = store.count_chunks()
        if actual != expected:
            raise MigrationError(f"迁移计数不一致: {actual} != {expected}")
        os.replace(temporary, target)
        return target
    except Exception as exc:
        raise MigrationError(f"旧知识库迁移失败，原文件保留于 {source}") from exc
