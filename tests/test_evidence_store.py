import importlib.util
import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from graphrag.models import EvidenceChunk
from graphrag.storage import KnowledgeStore
from graphrag.structure import parse_text


def api():
    assert importlib.util.find_spec('graphrag.evidence_store') is not None
    from graphrag.evidence_store import EvidenceStore, inspect_index
    return EvidenceStore, inspect_index


def chunks(doc='doc'):
    return [replace(c, embedding=np.array([1., i + 1.], dtype=np.float32))
            for i, c in enumerate(parse_text('# 第一条 范围\n依照第二条办理，另见第九条。\n# 第二条 规则\n适用规则。', doc))]


def test_inspection_and_constructor_never_modify_old_or_missing(tmp_path):
    store_type, inspect = api()
    missing = tmp_path / 'missing.sqlite'
    assert not inspect(missing)['exists']
    assert not missing.exists()
    old = tmp_path / 'old.sqlite'
    KnowledgeStore(old)
    original = old.read_bytes()
    caps = inspect(old)
    assert caps['exists'] and caps['vector'] and not caps['evidence_ready']
    from graphrag.errors import IndexCapabilityError
    with pytest.raises(IndexCapabilityError):
        store_type(old)
    assert old.read_bytes() == original
    with pytest.raises(IndexCapabilityError):
        store_type(missing)
    assert not missing.exists()


def test_exclusive_creation_and_reopen_read_only(tmp_path):
    store_type, inspect = api()
    path = tmp_path / 'new.sqlite'
    store_type.create(path)
    original = path.read_bytes()
    caps = inspect(path)
    assert caps['evidence_ready'] and caps['lexical'] and caps['vector']
    assert not caps['legacy_graph'] and not caps['legacy_topic']
    store_type(path)
    assert original == path.read_bytes()
    with pytest.raises(FileExistsError):
        store_type.create(path)
    assert original == path.read_bytes()


def test_fts_failure_explicit_and_keeps_new_diagnostic_file(tmp_path, monkeypatch):
    store_type, inspect = api()
    import graphrag.evidence_store as module
    from graphrag.errors import IndexCapabilityError
    monkeypatch.setattr(module, 'FTS_SCHEMA', 'CREATE VIRTUAL TABLE chunk_fts USING absent_fts5(text);')
    path = tmp_path / 'new.sqlite'
    with pytest.raises(IndexCapabilityError, match='FTS5'):
        store_type.create(path)
    assert path.exists() and not inspect(path)['evidence_ready']


def test_incomplete_hidden_completion_links_roundtrip_and_idempotence(tmp_path):
    store_type, _ = api()
    store = store_type.create(tmp_path / 'new.sqlite')
    records = chunks()
    store.prepare_document('doc', records, doc_name='文件', doc_path='doc.md')
    store.write_document_batch('doc', records[:1], 1, len(records))
    assert store.list_chunks() == []
    assert store.vector_matrix()[0] == []
    assert store.lexical_search('范围') == []
    assert store.links_for([records[0].id]) == []
    assert len(store.list_chunks(include_incomplete=True)) == 1
    assert store.list_documents()[0]['state'] == 'building'
    store.write_document_batch('doc', records[1:], len(records), len(records))
    restored = store.list_chunks()
    for actual, expected in zip(restored, records):
        assert {k: v for k, v in vars(actual).items() if k != 'embedding'} == {
            k: v for k, v in vars(expected).items() if k != 'embedding'}
        np.testing.assert_array_equal(actual.embedding, expected.embedding)
    links = store.links_for([records[0].id])
    assert any(l.kind == 'explicit_reference' and l.target_id == records[1].id for l in links)
    assert any(l.status == 'missing' for l in links)
    assert any(l.kind == 'section_member' for l in links)
    store.write_document_batch('doc', records[1:], len(records), len(records))
    assert store.links_for([records[0].id]) == links
    assert len(store.vector_matrix()[0]) == len(records)
    assert store.list_documents()[0]['state'] == 'complete'


def test_failure_rolls_back_fts_source_progress_and_preserves_cache(tmp_path, monkeypatch):
    store_type, _ = api()
    import graphrag.evidence_store as module
    store = store_type.create(tmp_path / 'new.sqlite')
    records = chunks()
    store.prepare_document('doc', records)
    store.write_document_batch('doc', records[:1], 1, 2)
    store.vector_matrix()
    cached = store.vector_matrix()
    def fail(_):
        raise RuntimeError('relationship failure')
    monkeypatch.setattr(module, 'build_links', fail)
    with pytest.raises(RuntimeError, match='relationship failure'):
        store.write_document_batch('doc', records[1:], 2, 2)
    assert store.get_progress('doc') == 1
    assert store.vector_matrix() is cached
    assert len(store.list_chunks(include_incomplete=True)) == 1
    with sqlite3.connect(store.path) as conn:
        assert conn.execute('SELECT count(*) FROM chunk_sources').fetchone()[0] == 1
        assert conn.execute('SELECT count(*) FROM chunk_fts').fetchone()[0] == 1


def test_resume_rejects_changed_unfinished_suffix_and_written_prefix(tmp_path):
    store_type, _ = api()
    store = store_type.create(tmp_path / 'new.sqlite')
    records = chunks()
    store.prepare_document('doc', records)
    store.write_document_batch('doc', records[:1], 1, 2)
    store.validate_source('doc', records)
    altered = [records[0], replace(records[1], source_text='changed suffix')]
    with pytest.raises(ValueError, match='source'):
        store.validate_source('doc', altered)
    with pytest.raises(ValueError, match='source'):
        store.prepare_document('doc', altered)
    with pytest.raises(ValueError, match='source'):
        store.write_document_batch('doc', [replace(records[0], clean_text='changed')], 1, 2)
    assert store.get_progress('doc') == 1


def test_lexical_bound_literals_order_limit_scope_and_links_scope(tmp_path):
    store_type, _ = api()
    store = store_type.create(tmp_path / 'new.sqlite')
    a, b = chunks('a'), chunks('b')
    for doc, values in [('a', a), ('b', b)]:
        store.write_document_batch(doc, values, len(values), len(values))
    assert store.lexical_search('...') == []
    assert store.lexical_search('" OR * : NOT NEAR(范围)')
    result = store.lexical_search('范围', doc_id='a', limit=1)
    assert len(result) == 1 and result[0][0] == a[0].id
    assert store.lexical_search('范围') == sorted(store.lexical_search('范围'), key=lambda row: (row[1], row[0]))
    assert store.links_for([b[0].id], doc_id='a') == []
    assert all(l.source_id == a[0].id for l in store.links_for([a[0].id, b[0].id], doc_id='a'))
    assert store.list_chunks('a', [b[0].id]) == []


def test_invalid_batches_cannot_complete_gaps_or_overwrite_other_document(tmp_path):
    store_type, _ = api()
    store = store_type.create(tmp_path / 'new.sqlite')
    records = chunks()
    with pytest.raises(ValueError):
        store.write_document_batch('doc', records[1:], 2, 2)
    assert store.list_documents() == []
    with pytest.raises(ValueError):
        store.write_document_batch('wrong', records, 2, 2)
    with pytest.raises(ValueError, match='prepare_document'):
        store.write_document_batch('doc', records[:1], 1, 2)
    assert store.count_chunks() == 0


def test_inspection_rejects_incomplete_source_schema_without_repair(tmp_path):
    store_type, inspect = api()
    path = tmp_path / 'new.sqlite'
    store_type.create(path)
    with sqlite3.connect(path) as conn:
        conn.execute('DROP TABLE chunk_sources')
        conn.execute('CREATE TABLE chunk_sources(chunk_id TEXT PRIMARY KEY)')
    original = path.read_bytes()
    assert not inspect(path)['evidence_ready']
    from graphrag.errors import IndexCapabilityError
    with pytest.raises(IndexCapabilityError):
        store_type(path)
    assert path.read_bytes() == original


def test_failed_embedding_write_and_replay_do_not_damage_other_document(tmp_path):
    store_type, _ = api()
    store = store_type.create(tmp_path / 'new.sqlite')
    a, b = chunks('a'), chunks('b')
    store.write_document_batch('a', a, 2, 2)
    store.vector_matrix()
    cached = store.vector_matrix()
    with pytest.raises(ValueError, match='dimensions'):
        store.write_document_batch('b', [b[0], replace(b[1], embedding=[1., 2., 3.])], 2, 2)
    assert store.list_documents()[0]['id'] == 'a'
    assert len(store.list_documents()) == 1
    assert store.vector_matrix() is cached
    with pytest.raises(ValueError, match='another document'):
        store.write_document_batch('b', [replace(b[0], id=a[0].id), b[1]], 2, 2)
    assert store.count_chunks() == 2
    store.write_document_batch('b', b, 2, 2)
    assert len(store.vector_matrix()[0]) == 4
    assert store.vector_matrix() is not cached


def test_parser_gap_offsets_and_partial_flags_round_trip(tmp_path):
    store_type, _ = api()
    store = store_type.create(tmp_path / 'new.sqlite')
    records = [replace(c, embedding=[1., 1.]) for c in parse_text(
        '# 第一条\r\n第一句。\r\n第二句。\r\n一个非常长的句子需要切分。', 'doc', chunk_size=7)]
    assert any(c.source_gap_before for c in records)
    assert any(c.partial for c in records)
    store.write_document_batch('doc', records, len(records), len(records))
    for actual, expected in zip(store.list_chunks(), records):
        assert actual.source_gap_before == expected.source_gap_before
        assert actual.heading_end == expected.heading_end
        assert actual.source_start == expected.source_start
        assert actual.source_end == expected.source_end
        assert actual.partial == expected.partial


@pytest.mark.parametrize(('next_index', 'total_chunks'), [(True, 2), (2, True), (2., 2), (2, 2.), (-1, 2)])
def test_batch_progress_requires_nonnegative_integers(tmp_path, next_index, total_chunks):
    store_type, _ = api()
    store = store_type.create(tmp_path / 'new.sqlite')
    with pytest.raises(ValueError, match='progress'):
        store.write_document_batch('doc', chunks(), next_index, total_chunks)
    assert store.list_documents() == []
