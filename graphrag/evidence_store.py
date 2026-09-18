"""Explicitly created evidence indexes; opening an index never migrates it.

Call prepare_document with *all* parsed chunks before embedding or resuming a
multi-batch document. It records ordinary source metadata, including the unseen
suffix, so later resumes can reject changed input. A single complete batch can
prepare its source records in the same transaction. No source hashes are used.
"""

from dataclasses import fields
import json
from pathlib import Path
import sqlite3

import numpy as np

from .errors import IndexCapabilityError
from .lexical import fts_index_text, fts_query
from .models import EvidenceChunk, EvidenceLink
from .storage import KnowledgeStore, SCHEMA, _ClosingConnection
from .structure import build_links


EVIDENCE_SCHEMA = """
ALTER TABLE documents ADD COLUMN state TEXT NOT NULL DEFAULT 'building';
ALTER TABLE documents ADD COLUMN total_chunks INTEGER NOT NULL DEFAULT 0;
CREATE TABLE parsed_sources(
  doc_id TEXT NOT NULL, sequence INTEGER NOT NULL, metadata TEXT NOT NULL,
  PRIMARY KEY(doc_id, sequence)
);
CREATE TABLE chunk_sources(chunk_id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
CREATE TABLE evidence_links(
  doc_id TEXT NOT NULL, source_id TEXT NOT NULL, target_id TEXT,
  kind TEXT NOT NULL, basis TEXT NOT NULL, status TEXT NOT NULL,
  source_start INTEGER NOT NULL, source_end INTEGER NOT NULL,
  target_ids TEXT NOT NULL
);
CREATE INDEX idx_evidence_links_source ON evidence_links(source_id);
"""
FTS_SCHEMA = """
CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61');
"""


def _readonly(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True,
                                 factory=_ClosingConnection)
    connection.row_factory = sqlite3.Row
    return connection


def inspect_index(path: str | Path) -> dict[str, object]:
    """Inspect capabilities through a read-only SQLite URI, without creating files."""
    path = Path(path)
    result = dict(exists=path.exists(), schema_version=None, vector=False,
                  lexical=False, evidence_ready=False, legacy_graph=False,
                  legacy_topic=False, reason='Index does not exist; create a new index.')
    if not result['exists']:
        return result
    try:
        with _readonly(path) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            metadata = dict(connection.execute('SELECT key, value FROM metadata')) if 'metadata' in tables else {}
            result['schema_version'] = metadata.get('schema_version')
            columns = {row[1] for row in connection.execute('PRAGMA table_info(chunks)')}
            result['vector'] = {'id', 'doc_id', 'sequence', 'embedding', 'embedding_dim',
                                'clean_text', 'context_text', 'chapter'} <= columns
            kind = metadata.get('index_kind')
            if kind != 'evidence':
                result['legacy_graph'] = 'concept_edges' in tables
                result['legacy_topic'] = {'topics', 'topic_chunks'} <= tables
            if 'chunk_fts' in tables:
                try:
                    connection.execute('SELECT chunk_id FROM chunk_fts LIMIT 0').fetchall()
                    result['lexical'] = True
                except sqlite3.DatabaseError:
                    pass
            required_columns = {
                'documents': {'id', 'name', 'path', 'state', 'total_chunks', 'processed_chunks'},
                'parsed_sources': {'doc_id', 'sequence', 'metadata'},
                'chunk_sources': {'chunk_id', 'metadata'},
                'evidence_links': {'doc_id', 'source_id', 'target_id', 'kind', 'basis',
                                   'status', 'source_start', 'source_end', 'target_ids'},
            }
            source_schema = all(
                required <= {row[1] for row in connection.execute(f'PRAGMA table_info({table})')}
                for table, required in required_columns.items())
            result['evidence_ready'] = bool(
                kind == 'evidence' and result['schema_version'] == '2' and result['vector']
                and result['lexical'] and source_schema)
            result['reason'] = '' if result['evidence_ready'] else (
                'Missing evidence index capabilities or unsupported version; explicitly rebuild from source into a new file.')
    except (sqlite3.DatabaseError, OSError) as exc:
        result['reason'] = f'Cannot inspect index: {exc}'
    return result


def _source_metadata(chunk):
    return json.dumps({field.name: getattr(chunk, field.name)
                       for field in fields(EvidenceChunk) if field.name != 'embedding'},
                      ensure_ascii=False, sort_keys=True)


def _restore(metadata, embedding):
    values = json.loads(metadata)
    for key in ('title_path', 'concepts'):
        values[key] = tuple(values[key])
    return EvidenceChunk(**values, embedding=np.frombuffer(embedding, dtype=np.float32).copy())


class EvidenceStore:
    """Evidence-only writes with compatible base vectors and one existing cache."""

    _vector_blob = staticmethod(KnowledgeStore._vector_blob)
    vector_matrix = KnowledgeStore.vector_matrix
    get_progress = KnowledgeStore.get_progress
    count_chunks = KnowledgeStore.count_chunks

    def __init__(self, path: str | Path):
        capabilities = inspect_index(path)
        if not capabilities['evidence_ready']:
            raise IndexCapabilityError(str(capabilities['reason']))
        self.path = Path(path)
        self._vector_cache = {}

    @classmethod
    def create(cls, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive file creation rejects even existing empty files.
        with path.open('xb'):
            pass
        with sqlite3.connect(path, factory=_ClosingConnection) as connection:
            connection.executescript(SCHEMA + EVIDENCE_SCHEMA)
            try:
                connection.executescript(FTS_SCHEMA)
            except sqlite3.DatabaseError as exc:
                raise IndexCapabilityError(
                    'SQLite FTS5 is required. The new partial index is retained for diagnosis; use a new output path after fixing FTS5.') from exc
            connection.executemany('INSERT INTO metadata(key,value) VALUES(?,?)',
                                   [('schema_version', '2'), ('index_kind', 'evidence')])
        return cls(path)

    def _connect(self):
        # Never recreate a removed index through SQLite's default create mode.
        connection = sqlite3.connect(self.path.resolve().as_uri() + '?mode=rw', uri=True,
                                     factory=_ClosingConnection)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _validate_records(doc_id, records):
        if any(not isinstance(c, EvidenceChunk) or c.doc_id != doc_id for c in records):
            raise ValueError('Chunks must belong to the specified document and have source metadata')
        if len({c.id for c in records}) != len(records):
            raise ValueError('Duplicate chunk IDs')

    def _prepare(self, connection, doc_id, chunks, doc_name, doc_path):
        self._validate_records(doc_id, chunks)
        if [c.sequence for c in chunks] != list(range(len(chunks))):
            raise ValueError('Full source chunks must have consecutive sequence numbers')
        existing = connection.execute('SELECT * FROM documents WHERE id=?', (doc_id,)).fetchone()
        serialized = [_source_metadata(c) for c in chunks]
        if existing:
            saved = [row[0] for row in connection.execute(
                'SELECT metadata FROM parsed_sources WHERE doc_id=? ORDER BY sequence', (doc_id,))]
            if existing['total_chunks'] != len(chunks) or saved != serialized:
                raise ValueError('Document source changed; rebuild into a new index')
            if ((doc_name and doc_name != existing['name']) or
                    (doc_path and doc_path != existing['path'])):
                raise ValueError('Document source name/path changed; rebuild into a new index')
            return
        connection.execute('INSERT INTO documents(id,name,path,total_chunks) VALUES(?,?,?,?)',
                           (doc_id, doc_name, doc_path, len(chunks)))
        connection.executemany('INSERT INTO parsed_sources(doc_id,sequence,metadata) VALUES(?,?,?)',
                               [(doc_id, c.sequence, metadata) for c, metadata in zip(chunks, serialized)])

    def prepare_document(self, doc_id, all_parsed_chunks, *, doc_name='', doc_path=''):
        """Persist or validate all source metadata before embedding/resuming."""
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            self._prepare(connection, doc_id, all_parsed_chunks, doc_name, doc_path)

    def validate_source(self, doc_id, all_parsed_chunks):
        self._validate_records(doc_id, all_parsed_chunks)
        with _readonly(self.path) as connection:
            document = connection.execute('SELECT total_chunks FROM documents WHERE id=?', (doc_id,)).fetchone()
            saved = [row[0] for row in connection.execute(
                'SELECT metadata FROM parsed_sources WHERE doc_id=? ORDER BY sequence', (doc_id,))]
        if document is None or document[0] != len(all_parsed_chunks) or saved != [_source_metadata(c) for c in all_parsed_chunks]:
            raise ValueError('Document source changed or was not prepared; rebuild into a new index')

    def write_document_batch(self, doc_id, chunks, next_index, total_chunks, *, doc_name='', doc_path=''):
        """Commit chunks, FTS and progress atomically; resolve links at completion."""
        self._validate_records(doc_id, chunks)
        if (type(next_index) is not int or type(total_chunks) is not int
                or not 0 <= next_index <= total_chunks
                or [c.sequence for c in chunks] != list(range(next_index - len(chunks), next_index))):
            raise ValueError('Batch sequence/progress is invalid')
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            document = connection.execute('SELECT * FROM documents WHERE id=?', (doc_id,)).fetchone()
            if document is None:
                if len(chunks) != total_chunks or next_index != total_chunks:
                    raise ValueError('Call prepare_document with the full source before multi-batch writes')
                self._prepare(connection, doc_id, chunks, doc_name, doc_path)
                document = connection.execute('SELECT * FROM documents WHERE id=?', (doc_id,)).fetchone()
            if document['total_chunks'] != total_chunks:
                raise ValueError('Document source chunk count changed')
            if ((doc_name and document['name'] != doc_name) or (doc_path and document['path'] != doc_path)):
                raise ValueError('Document source name/path changed')
            start_index = next_index - len(chunks)
            progress = document['processed_chunks']
            if start_index > progress or (start_index < progress < next_index):
                raise ValueError('Batch must continue progress or replay a completed batch')
            dimension = connection.execute('SELECT embedding_dim FROM chunks LIMIT 1').fetchone()
            for chunk in chunks:
                metadata = _source_metadata(chunk)
                expected = connection.execute('SELECT metadata FROM parsed_sources WHERE doc_id=? AND sequence=?',
                                              (doc_id, chunk.sequence)).fetchone()
                if expected is None or expected[0] != metadata:
                    raise ValueError('Document source changed; rebuild into a new index')
                vector = np.asarray(chunk.embedding, dtype=np.float32)
                if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                    raise ValueError('Embedding must be a nonempty finite vector')
                blob, size = self._vector_blob(vector)
                if dimension is not None and dimension[0] != size:
                    raise ValueError('Embedding dimensions must match the index')
                dimension = (size,)
                old = connection.execute('SELECT doc_id,sequence FROM chunks WHERE id=?', (chunk.id,)).fetchone()
                if old:
                    if old['doc_id'] != doc_id or old['sequence'] != chunk.sequence:
                        raise ValueError('Chunk ID belongs to another document or sequence')
                    continue
                connection.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)',
                                   (chunk.id, doc_id, chunk.sequence, chunk.clean_text, chunk.context_text,
                                    chunk.chapter, blob, size))
                connection.execute('INSERT INTO chunk_sources VALUES(?,?)', (chunk.id, metadata))
                connection.execute('INSERT INTO chunk_fts(chunk_id,text) VALUES(?,?)',
                                   (chunk.id, fts_index_text(chunk.clean_text)))
                connection.executemany('INSERT INTO chunk_concepts VALUES(?,?)',
                                       [(chunk.id, concept) for concept in dict.fromkeys(chunk.concepts)])
            new_progress = max(progress, next_index)
            count = connection.execute('SELECT count(*) FROM chunks WHERE doc_id=?', (doc_id,)).fetchone()[0]
            if count != new_progress:
                raise ValueError('Document has missing or conflicting chunk sequences')
            state = 'building'
            if new_progress == total_chunks:
                links = build_links(self._list_chunks(connection, doc_id, None, True))
                connection.execute('DELETE FROM evidence_links WHERE doc_id=?', (doc_id,))
                connection.executemany('INSERT INTO evidence_links VALUES(?,?,?,?,?,?,?,?,?)',
                    [(doc_id, l.source_id, l.target_id, l.kind, l.basis, l.status, l.source_start,
                      l.source_end, json.dumps(l.target_ids)) for l in links])
                state = 'complete'
            connection.execute('UPDATE documents SET processed_chunks=?,state=? WHERE id=?',
                               (new_progress, state, doc_id))
        self._vector_cache.clear()

    def list_documents(self):
        with _readonly(self.path) as connection:
            return [dict(row) for row in connection.execute('SELECT * FROM documents ORDER BY id')]

    @staticmethod
    def _list_chunks(connection, doc_id, chunk_ids, include_incomplete):
        clauses, params = [], []
        if not include_incomplete:
            clauses.append("d.state='complete'")
        if doc_id is not None:
            clauses.append('c.doc_id=?')
            params.append(doc_id)
        if chunk_ids is not None:
            if not chunk_ids:
                return []
            clauses.append('c.id IN (' + ','.join('?' for _ in chunk_ids) + ')')
            params.extend(chunk_ids)
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        rows = connection.execute('SELECT s.metadata,c.embedding FROM chunks c '
                                  'JOIN chunk_sources s ON s.chunk_id=c.id JOIN documents d ON d.id=c.doc_id'
                                  + where + ' ORDER BY c.doc_id,c.sequence', params)
        return [_restore(row['metadata'], row['embedding']) for row in rows]

    def list_chunks(self, doc_id=None, chunk_ids=None, *, include_incomplete=False):
        with _readonly(self.path) as connection:
            return self._list_chunks(connection, doc_id, chunk_ids, include_incomplete)

    def lexical_search(self, query, doc_id=None, limit=40):
        """Return (chunk ID, raw BM25) pairs, ascending score with ID tie breaks."""
        expression = fts_query(query)
        if not expression or limit <= 0:
            return []
        params = [expression]
        scope = ''
        if doc_id is not None:
            scope = ' AND c.doc_id=?'
            params.append(doc_id)
        params.append(limit)
        with _readonly(self.path) as connection:
            rows = connection.execute(
                'SELECT c.id,bm25(chunk_fts) AS score FROM chunk_fts '
                'JOIN chunks c ON c.id=chunk_fts.chunk_id JOIN documents d ON d.id=c.doc_id '
                "WHERE chunk_fts MATCH ? AND d.state='complete'" + scope + ' ORDER BY score,c.id LIMIT ?', params)
            return [(row[0], float(row[1])) for row in rows]

    def links_for(self, source_ids, doc_id=None):
        """Read local outgoing links only, including unresolved references."""
        if not source_ids:
            return []
        params = list(source_ids)
        scope = ''
        if doc_id is not None:
            scope = ' AND l.doc_id=?'
            params.append(doc_id)
        with _readonly(self.path) as connection:
            rows = connection.execute(
                'SELECT l.* FROM evidence_links l JOIN chunks c ON c.id=l.source_id AND c.doc_id=l.doc_id '
                'JOIN documents d ON d.id=l.doc_id '
                "WHERE d.state='complete' AND l.source_id IN (" + ','.join('?' for _ in source_ids) + ')' + scope +
                ' ORDER BY l.source_id,l.kind,l.source_start,l.target_id', params)
            return [EvidenceLink(row['source_id'], row['target_id'], row['kind'], row['basis'], row['status'],
                                 row['source_start'], row['source_end'], tuple(json.loads(row['target_ids']))) for row in rows]
