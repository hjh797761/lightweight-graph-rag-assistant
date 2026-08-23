from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sqlite3

import numpy as np

from .models import ChunkRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL DEFAULT '',
  path TEXT NOT NULL DEFAULT '',
  processed_chunks INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks(
  id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  sequence INTEGER NOT NULL,
  clean_text TEXT NOT NULL,
  context_text TEXT NOT NULL,
  chapter TEXT NOT NULL,
  embedding BLOB NOT NULL,
  embedding_dim INTEGER NOT NULL,
  UNIQUE(doc_id, sequence)
);
CREATE TABLE IF NOT EXISTS chunk_concepts(
  chunk_id TEXT NOT NULL,
  concept TEXT NOT NULL,
  PRIMARY KEY(chunk_id, concept)
);
CREATE TABLE IF NOT EXISTS concept_edges(
  source TEXT NOT NULL,
  target TEXT NOT NULL,
  weight INTEGER NOT NULL,
  PRIMARY KEY(source, target)
);
CREATE TABLE IF NOT EXISTS topics(
  id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  title TEXT NOT NULL,
  centroid BLOB NOT NULL,
  embedding_dim INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS topic_chunks(
  topic_id TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  PRIMARY KEY(topic_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_sequence ON chunks(doc_id, sequence);
CREATE INDEX IF NOT EXISTS idx_chunk_concepts_concept ON chunk_concepts(concept);
"""


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class KnowledgeStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._vector_cache: dict[str | None, tuple[list[str], np.ndarray]] = {}
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES('schema_version', '1')"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, factory=_ClosingConnection)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _vector_blob(vector) -> tuple[bytes, int]:
        value = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        return value.tobytes(), int(value.size)

    def ensure_document(self, doc_id: str, name: str = "", path: str = "") -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents(id, name, path) VALUES(?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  name=CASE WHEN excluded.name='' THEN documents.name ELSE excluded.name END,
                  path=CASE WHEN excluded.path='' THEN documents.path ELSE excluded.path END
                """,
                (doc_id, name, path),
            )

    def write_chunk_batch(
        self,
        doc_id: str,
        chunks: list[ChunkRecord],
        next_index: int,
        *,
        doc_name: str = "",
        doc_path: str = "",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents(id, name, path, processed_chunks) VALUES(?, ?, ?, 0)
                ON CONFLICT(id) DO UPDATE SET
                  name=CASE WHEN excluded.name='' THEN documents.name ELSE excluded.name END,
                  path=CASE WHEN excluded.path='' THEN documents.path ELSE excluded.path END
                """,
                (doc_id, doc_name, doc_path),
            )
            for record in chunks:
                blob, dimension = self._vector_blob(record.embedding)
                connection.execute(
                    """
                    INSERT INTO chunks(
                      id, doc_id, sequence, clean_text, context_text, chapter,
                      embedding, embedding_dim
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                      clean_text=excluded.clean_text,
                      context_text=excluded.context_text,
                      chapter=excluded.chapter,
                      embedding=excluded.embedding,
                      embedding_dim=excluded.embedding_dim
                    """,
                    (
                        record.id,
                        record.doc_id,
                        record.sequence,
                        record.clean_text,
                        record.context_text,
                        record.chapter,
                        blob,
                        dimension,
                    ),
                )
                connection.execute(
                    "DELETE FROM chunk_concepts WHERE chunk_id=?", (record.id,)
                )
                unique_concepts = tuple(dict.fromkeys(record.concepts))
                connection.executemany(
                    "INSERT INTO chunk_concepts(chunk_id, concept) VALUES(?, ?)",
                    [(record.id, concept) for concept in unique_concepts],
                )
                for source in unique_concepts:
                    for target in unique_concepts:
                        if source == target:
                            continue
                        connection.execute(
                            """
                            INSERT INTO concept_edges(source, target, weight) VALUES(?, ?, 1)
                            ON CONFLICT(source, target) DO UPDATE SET weight=weight + 1
                            """,
                            (source, target),
                        )
            connection.execute(
                "UPDATE documents SET processed_chunks=? WHERE id=?",
                (next_index, doc_id),
            )
        self._vector_cache.clear()

    def set_progress(self, doc_id: str, next_index: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO documents(id) VALUES(?)", (doc_id,)
            )
            connection.execute(
                "UPDATE documents SET processed_chunks=? WHERE id=?",
                (next_index, doc_id),
            )

    def get_progress(self, doc_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT processed_chunks FROM documents WHERE id=?", (doc_id,)
            ).fetchone()
        return int(row[0]) if row else 0

    def count_chunks(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def list_documents(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name, path, processed_chunks FROM documents ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_chunks(
        self, doc_id: str | None = None, chunk_ids: list[str] | None = None
    ) -> list[ChunkRecord]:
        clauses: list[str] = []
        parameters: list[object] = []
        if doc_id:
            clauses.append("c.doc_id=?")
            parameters.append(doc_id)
        if chunk_ids is not None:
            if not chunk_ids:
                return []
            clauses.append(f"c.id IN ({','.join('?' for _ in chunk_ids)})")
            parameters.extend(chunk_ids)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT c.*, GROUP_CONCAT(cc.concept, char(31)) AS concepts
                FROM chunks c
                LEFT JOIN chunk_concepts cc ON cc.chunk_id=c.id
                {where}
                GROUP BY c.id
                ORDER BY c.doc_id, c.sequence
                """,
                parameters,
            ).fetchall()
        records = []
        for row in rows:
            concepts = tuple(row["concepts"].split(chr(31))) if row["concepts"] else ()
            records.append(
                ChunkRecord(
                    id=row["id"],
                    doc_id=row["doc_id"],
                    sequence=int(row["sequence"]),
                    clean_text=row["clean_text"],
                    context_text=row["context_text"],
                    chapter=row["chapter"],
                    concepts=concepts,
                    embedding=np.frombuffer(row["embedding"], dtype=np.float32).copy(),
                )
            )
        return records

    def vector_matrix(self, doc_id: str | None = None) -> tuple[list[str], np.ndarray]:
        if doc_id in self._vector_cache:
            return self._vector_cache[doc_id]
        chunks = self.list_chunks(doc_id)
        ids = [chunk.id for chunk in chunks]
        if chunks:
            matrix = np.vstack([chunk.embedding for chunk in chunks]).astype(
                np.float32, copy=False
            )
        else:
            matrix = np.empty((0, 0), dtype=np.float32)
        self._vector_cache[doc_id] = (ids, matrix)
        return ids, matrix

    def load_graph(self) -> dict[str, dict[str, int]]:
        graph: dict[str, dict[str, int]] = defaultdict(dict)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source, target, weight FROM concept_edges"
            ).fetchall()
        for row in rows:
            graph[row["source"]][row["target"]] = int(row["weight"])
        return dict(graph)

    def replace_topics(self, doc_id: str, topics: list[dict[str, object]]) -> None:
        with self._connect() as connection:
            old_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT id FROM topics WHERE doc_id=?", (doc_id,)
                ).fetchall()
            ]
            for topic_id in old_ids:
                connection.execute(
                    "DELETE FROM topic_chunks WHERE topic_id=?", (topic_id,)
                )
            connection.execute("DELETE FROM topics WHERE doc_id=?", (doc_id,))
            for topic in topics:
                blob, dimension = self._vector_blob(topic["centroid"])
                connection.execute(
                    "INSERT INTO topics(id, doc_id, title, centroid, embedding_dim) VALUES(?, ?, ?, ?, ?)",
                    (topic["id"], doc_id, topic["title"], blob, dimension),
                )
                connection.executemany(
                    "INSERT INTO topic_chunks(topic_id, chunk_id) VALUES(?, ?)",
                    [(topic["id"], chunk_id) for chunk_id in topic.get("chunk_ids", [])],
                )

    def list_topics(self, doc_id: str | None = None) -> list[dict[str, object]]:
        where = "WHERE t.doc_id=?" if doc_id else ""
        params = (doc_id,) if doc_id else ()
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT t.*, GROUP_CONCAT(tc.chunk_id, char(31)) AS chunk_ids
                FROM topics t LEFT JOIN topic_chunks tc ON tc.topic_id=t.id
                {where} GROUP BY t.id ORDER BY t.id
                """,
                params,
            ).fetchall()
        return [
            {
                "id": row["id"],
                "doc_id": row["doc_id"],
                "title": row["title"],
                "centroid": np.frombuffer(row["centroid"], dtype=np.float32).copy(),
                "chunk_ids": row["chunk_ids"].split(chr(31)) if row["chunk_ids"] else [],
            }
            for row in rows
        ]

    def import_legacy(self, data: dict) -> None:
        documents = data.get("documents", {})
        chunks_by_doc: dict[str, list[ChunkRecord]] = defaultdict(list)
        for position, (chunk_id, raw) in enumerate(data.get("chunks", {}).items()):
            doc_id = str(raw.get("doc_id") or "legacy")
            try:
                sequence = int(chunk_id.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                sequence = position
            chunks_by_doc[doc_id].append(
                ChunkRecord(
                    id=str(raw.get("id") or chunk_id),
                    doc_id=doc_id,
                    sequence=sequence,
                    clean_text=str(raw.get("clean_text") or raw.get("text") or ""),
                    context_text=str(raw.get("text") or raw.get("clean_text") or ""),
                    chapter=str(raw.get("chapter") or "未分类"),
                    concepts=tuple(raw.get("concepts") or ()),
                    embedding=np.asarray(raw.get("embedding") or [], dtype=np.float32),
                )
            )
        progress = data.get("processed_chunks", {})
        for doc_id, chunks in chunks_by_doc.items():
            chunks.sort(key=lambda item: item.sequence)
            metadata = documents.get(doc_id, {})
            self.write_chunk_batch(
                doc_id,
                chunks,
                next_index=int(progress.get(doc_id, len(chunks))),
                doc_name=str(metadata.get("name") or ""),
                doc_path=str(metadata.get("path") or ""),
            )
