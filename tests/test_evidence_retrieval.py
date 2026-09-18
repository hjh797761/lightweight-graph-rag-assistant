"""CPU integration coverage for bounded evidence retrieval and provenance."""

from dataclasses import replace
import importlib.util
import math

import numpy as np
import pytest

from graphrag.evidence_store import EvidenceStore
from graphrag.models import EvidenceChunk, EvidenceOptions, RetrievalResult


def api():
    assert importlib.util.find_spec('graphrag.evidence_retrieval'), 'EvidenceRetriever is missing'
    from graphrag.evidence_retrieval import EvidenceRetriever
    return EvidenceRetriever


class Embedder:
    def __init__(self, vector=(1., 0.)):
        self.vector = np.asarray(vector)
        self.calls = []

    def encode_one(self, query):
        self.calls.append(query)
        return self.vector


class Reranker:
    def __init__(self, transform=None):
        self.calls = []
        self.transform = transform

    def __call__(self, query, chunks):
        self.calls.append((query, chunks))
        return self.transform(chunks) if self.transform else [(c, 1.) for c in chunks]


def chunk(id, text='ordinary text', *, doc='doc', sequence=0, vector=(1., 0.), section=None, title=()):
    return EvidenceChunk(id, doc, sequence, text, text, embedding=np.array(vector),
                         section_id=section, title_path=title, locator=f'L{sequence+1}',
                         source_start=sequence * 100, source_end=sequence * 100 + len(text), source_text=text)


def write(store, records, *, name='rules.md', path='/docs/rules.md'):
    store.write_document_batch(records[0].doc_id, records, len(records), len(records),
                               doc_name=name, doc_path=path)


@pytest.fixture
def store(tmp_path):
    return EvidenceStore.create(tmp_path / 'evidence.db')


def test_direct_rule_cross_section_reference_and_local_one_hop(store):
    Retriever = api()
    records = [chunk('rule', '申请规则见第二条。', section='s1', title=('第一条 申请',)),
               chunk('condition', '申请条件为年满十八岁。', sequence=1, vector=(0., 1.), section='s2', title=('第二条 条件',)),
               chunk('outside', '不相关说明。', sequence=2, vector=(0., 1.), section='s3', title=('第三条 其他',))]
    write(store, records)
    write(store, [chunk('other', '申请规则见第二条。', doc='other', vector=(-1., 0.))], name='other.md')
    reranker = Reranker()
    options = EvidenceOptions(candidate_k=1, lexical_k=1, max_chunks=2, max_supplements=1)
    result = Retriever(store, Embedder(), reranker=reranker, options=options).retrieve('申请规则', doc_scope='doc')
    assert [c.id for c in result.chunks] == ['rule', 'condition']
    assert result.selection.roles == ['core', 'supplement']
    assert result.scores == [1., None]
    assert '申请条件为年满十八岁' in result.context and '引用目标' in result.context
    assert len(reranker.calls[0][1]) == 1
    assert result.selection.budget_used == len(result.context) <= options.context_budget
    assert result.candidate_details[0]['final_role'] == 'core'
    support = next(d for d in result.candidate_details if d['chunk_id'] == 'condition')
    assert support['rerank_score'] is None and support['final_role'] == 'supplement'


def test_scope_exact_ids_names_paths_and_ambiguity(store):
    Retriever = api()
    write(store, [chunk('a', doc='first')], path='/one/rules.md')
    write(store, [chunk('b', doc='second')], path='/two/rules.md')
    engine = Retriever(store, Embedder())
    for scope in ['rules.md', 'rules', 'unknown']:
        with pytest.raises(ValueError, match='ambiguous|Unknown'):
            engine.retrieve('ordinary', doc_scope=scope)
    for scope in ['first', '/one/rules.md']:
        assert [c.id for c in engine.retrieve('ordinary', doc_scope=scope).chunks] == ['a']
    write(store, [chunk('c', doc='rules.md')], name='third.md')
    assert [c.id for c in engine.retrieve('ordinary', doc_scope='rules.md').chunks] == ['c']


def test_empty_and_incomplete_never_embed(store):
    Retriever = api()
    embedder = Embedder()
    engine = Retriever(store, embedder)
    assert engine.retrieve('ordinary').selection.status == 'no_candidates'
    records = [chunk('a'), chunk('b', sequence=1)]
    store.prepare_document('doc', records)
    store.write_document_batch('doc', records[:1], 1, 2)
    assert engine.retrieve('ordinary').chunks == []
    with pytest.raises(ValueError, match='complete'):
        engine.retrieve('ordinary', doc_scope='doc')
    with pytest.raises(ValueError, match='Unknown'):
        engine.retrieve('', doc_scope='unknown')
    assert engine.retrieve('   ').selection.status == 'no_candidates'
    assert embedder.calls == []


@pytest.mark.parametrize('top_k', [0, -1, True, 1.5])
def test_invalid_top_k(store, top_k):
    with pytest.raises(ValueError, match='top_k'):
        api()(store, Embedder()).retrieve('', top_k=top_k)


def test_bounded_union_trace_rrf_ties_and_disabled_reranker(store):
    Retriever = api()
    write(store, [chunk('a', 'dense', vector=(1., 0.)),
                  chunk('b', 'dense', sequence=1, vector=(1., 0.)),
                  chunk('c', 'keyword', sequence=2, vector=(0., 1.)),
                  chunk('d', 'keyword', sequence=3, vector=(0., 1.))])
    options = EvidenceOptions(candidate_k=2, lexical_k=2, max_supplements=0)
    result = Retriever(store, Embedder(), options=options).retrieve('keyword', top_k=1)
    assert [d['chunk_id'] for d in result.candidate_details] == ['a', 'c', 'b', 'd']
    assert result.counts == dict(vector_candidates=2, lexical_candidates=2,
                                 candidate_union=4, fused_candidates=2, reranked_candidates=0)
    assert [c.id for c in result.chunks] == ['a']
    assert result.scores == [1 / 61]
    assert result.requested_top_k == 1
    assert all(d['rerank_score'] is None for d in result.candidate_details)
    assert result.candidate_details[2]['omission_reason'] == 'fusion_cap'
    assert result.candidate_details[0]['vector_score'] == 1.
    assert result.candidate_details[1]['lexical_score'] < 0
    assert result.stage_seconds and all(math.isfinite(v) and v >= 0 for v in result.stage_seconds.values())


@pytest.mark.parametrize('mode', ['unknown', 'duplicate', 'nan', 'bool'])
def test_malformed_reranker_rejected(store, mode):
    Retriever = api()
    write(store, [chunk('a')])
    def bad(chunks):
        c = chunks[0]
        return {'unknown': [(replace(c, id='injected'), 1.)], 'duplicate': [(c, 1.), (c, .2)],
                'nan': [(c, math.nan)], 'bool': [(c, True)]}[mode]
    with pytest.raises(ValueError, match='reranker'):
        Retriever(store, Embedder(), reranker=Reranker(bad)).retrieve('ordinary')


def test_reranker_authoritative_records_omissions_and_tie_order(store):
    Retriever = api()
    write(store, [chunk('a'), chunk('b', sequence=1), chunk('c', sequence=2)])
    reranker = Reranker(lambda chunks: [(replace(chunks[1], source_text='INJECTED'), .5), (chunks[0], .5)])
    result = Retriever(store, Embedder(), reranker=reranker,
                       options=EvidenceOptions(max_supplements=0)).retrieve('ordinary')
    assert [c.id for c in result.chunks] == ['a', 'b']
    assert 'INJECTED' not in result.context
    assert result.candidate_details[2]['omission_reason'] == 'reranker_omitted'
    assert result.counts['reranked_candidates'] == 3


def test_model_error_propagates(store):
    write(store, [chunk('a')])
    def fail(_):
        raise RuntimeError('model failed')
    with pytest.raises(RuntimeError, match='model failed'):
        api()(store, Embedder(), reranker=Reranker(fail)).retrieve('ordinary')


@pytest.mark.parametrize('vector', [(math.nan, 0.), (math.inf, 0.), (1.,), ((1., 0.),)])
def test_bad_query_vectors(store, vector):
    write(store, [chunk('a')])
    with pytest.raises(ValueError, match='vector|dimension'):
        api()(store, Embedder(vector)).retrieve('ordinary')


def test_budget_whole_render_and_no_structure(store):
    Retriever = api()
    write(store, [chunk('a', 'plain original text')])
    engine = Retriever(store, Embedder(), options=EvidenceOptions(max_supplements=0))
    baseline = engine.retrieve('original')
    assert any(d.get('association_state') == 'no_links' for d in baseline.selection.decisions)
    for budget in [baseline.selection.budget_used-1, baseline.selection.budget_used]:
        result = Retriever(store, Embedder(), options=EvidenceOptions(context_budget=budget, max_supplements=0)).retrieve('original')
        assert result.selection.budget_used == len(result.context) <= budget
        assert bool(result.chunks) == (budget == baseline.selection.budget_used)
    counter = lambda text: len(text.encode('utf8'))
    token = Retriever(store, Embedder(), options=EvidenceOptions(budget_unit='tokens'), count_tokens=counter).retrieve('original')
    assert token.selection.budget_used == counter(token.context)


def test_backward_compatible_result_and_independent_metadata():
    a = RetrievalResult('old', [], [], '', None, 6)
    assert hasattr(a, 'selection'), 'Result needs optional evidence metadata'
    b = RetrievalResult('other')
    a.candidate_details.append({'chunk_id': 'a'})
    a.stage_seconds['query'] = .1
    a.counts['candidate_union'] = 1
    assert b.candidate_details == [] and b.stage_seconds == {} and b.counts == {}
    assert a.selection is None


def test_candidate_materialization_is_bounded_and_membership_not_fetched(store, monkeypatch):
    Retriever = api()
    records = [chunk('a', 'alpha', section='section-a', title=('第一条 甲',)),
               chunk('b', 'beta', sequence=1, section='section-b', title=('第二条 乙',))]
    write(store, records)
    # The existing shared vector cache has its own cold-load implementation.
    # Subsequent source materialization by this retriever must remain local.
    store.vector_matrix()
    source_reads, link_reads = [], []
    original_chunks, original_links = store.list_chunks, store.links_for
    def read_chunks(doc_id=None, chunk_ids=None):
        assert chunk_ids is not None, 'Retriever must not scan all source records'
        source_reads.append(chunk_ids)
        return original_chunks(doc_id, chunk_ids)
    def read_links(source_ids, doc_id=None):
        link_reads.append(source_ids)
        return original_links(source_ids, doc_id)
    monkeypatch.setattr(store, 'list_chunks', read_chunks)
    monkeypatch.setattr(store, 'links_for', read_links)
    options = EvidenceOptions(candidate_k=1, lexical_k=1, max_chunks=2, max_supplements=1)
    result = Retriever(store, Embedder(), options=options).retrieve('alpha')
    assert source_reads == [['a']] and link_reads == [['a']]
    assert any(d.get('association_state') == 'section_membership_only' for d in result.selection.decisions)


def test_reference_fetches_every_target_and_supplement_can_overlap_pool(store):
    Retriever = api()
    records = [chunk('a', '规则见第二条', section='s1', title=('第一条 规则',)),
               chunk('b', '条件前半', sequence=1, section='s2', title=('第二条 条件',)),
               chunk('c', '条件后半', sequence=2, section='s2', title=('第二条 条件',))]
    write(store, records)
    options = EvidenceOptions(candidate_k=3, max_chunks=3, max_supplements=2)
    result = Retriever(store, Embedder(), reranker=Reranker(), options=options).retrieve('规则')
    assert [c.id for c in result.chunks] == ['a', 'b', 'c']
    assert result.selection.roles == ['core', 'supplement', 'supplement']
    assert result.scores == [1., None, None]
    assert all(d['rerank_score'] == 1. for d in result.candidate_details)
    assert len(result.selection.reasons['b']) == len(result.selection.reasons['c']) == 1


def test_unresolved_references_remain_diagnostic(store):
    write(store, [chunk('a', '规则见第九条', section='s1', title=('第一条 规则',))])
    result = api()(store, Embedder()).retrieve('规则')
    assert any(d.get('status') == 'missing' for d in result.selection.decisions)
    assert any(d.get('association_state') == 'unresolved_only' for d in result.selection.decisions)


def test_nonfinite_cached_matrix_rejected(store):
    write(store, [chunk('a')])
    _, matrix = store.vector_matrix()
    matrix[0, 0] = math.nan
    with pytest.raises(ValueError, match='vector'):
        api()(store, Embedder()).retrieve('ordinary')
