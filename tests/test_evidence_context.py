import math
from dataclasses import FrozenInstanceError

import pytest

from graphrag.models import EvidenceChunk, EvidenceLink


def api():
    from graphrag import models
    assert hasattr(models, "EvidenceOptions"), "EvidenceOptions has not been implemented"
    from graphrag.evidence_context import assemble_context
    return models.EvidenceOptions, assemble_context


def chunk(id, text="原文", *, start=0, doc="doc", title=(), gap=""):
    return EvidenceChunk(id, doc, 0, text, text, title_path=title, locator="L1",
                         source_start=start, source_end=start + len(text),
                         source_text=text, source_gap_before=gap)


def link(source, target, kind="explicit_reference", *, ids=(), status="resolved", basis="见附件"):
    return EvidenceLink(source, target, kind, basis=basis, status=status, target_ids=ids)


def test_options_defaults_and_frozen():
    Options, _ = api()
    options = Options()
    assert (options.candidate_k, options.lexical_k, options.rrf_k) == (40, 40, 60)
    assert (options.max_chunks, options.max_supplements, options.context_budget) == (6, 2, 6000)
    assert (options.supplement_fraction, options.budget_unit) == (.30, "characters")
    with pytest.raises(FrozenInstanceError):
        options.max_chunks = 3


@pytest.mark.parametrize("kwargs", [
    {"candidate_k": 0}, {"lexical_k": True}, {"rrf_k": 1.2}, {"max_chunks": 0},
    {"max_supplements": -1}, {"max_supplements": False}, {"context_budget": True},
    {"supplement_fraction": math.nan}, {"supplement_fraction": math.inf},
    {"supplement_fraction": -.1}, {"supplement_fraction": 1},
    {"supplement_fraction": True}, {"budget_unit": "words"},
])
def test_invalid_options(kwargs):
    Options, _ = api()
    with pytest.raises(ValueError):
        Options(**kwargs)


def test_empty_and_token_counter_required_even_when_empty():
    Options, assemble = api()
    def never(_):
        pytest.fail("empty results must not load links")
    result = assemble([], never, options=Options())
    assert result.context == "" and result.budget_used == 0 and result.status == "no_candidates"
    assert result.chunks == result.roles == result.scores == []
    with pytest.raises(ValueError, match="count_tokens"):
        assemble([], never, options=Options(budget_unit="tokens"))


def test_render_cost_includes_metadata_and_exact_token_counter():
    Options, assemble = api()
    core = chunk("a", "12345", title=("长标题" * 50,))
    result = assemble([(core, .5)], lambda _: [], options=Options(max_supplements=0))
    assert result.budget_used == len(result.context) > len(core.source_text)
    assert result.scores == [.5] and result.roles == ["core"]
    exact = assemble([(core, .5)], lambda _: [], options=Options(max_supplements=0, context_budget=len(result.context)))
    assert exact.context == result.context
    small = assemble([(core, .5)], lambda _: [], options=Options(context_budget=len(result.context)-1, max_supplements=0))
    assert small.status == "budget_exhausted" and small.context == ""
    counter = lambda text: len(text.encode("utf-8"))
    token = assemble([(core, .5)], lambda _: [], options=Options(budget_unit="tokens"), count_tokens=counter)
    assert token.budget_used == counter(token.context) and token.unit == "tokens"


@pytest.mark.parametrize("kwargs", [{"max_chunks": 1}, {"max_supplements": 0}, {"supplement_fraction": 0}])
def test_disabled_supplements(kwargs):
    Options, assemble = api()
    a, b = chunk("a"), chunk("b", start=20)
    result = assemble([(a, .9)], lambda _: [(link("a", "b"), [b])], options=Options(**kwargs))
    assert [c.id for c in result.chunks] == ["a"]


def test_one_hop_priority_shared_reasons_and_null_supplement_score():
    Options, assemble = api()
    a, b, shared, nearby, second = [chunk(x, x * 3, start=i*20) for i, x in enumerate("abscz")]
    calls = []
    def loader(ids):
        calls.append(ids)
        return [(link("a", "c", "adjacent"), [nearby]),
                (link("b", "s", basis="引用乙"), [shared]),
                (link("a", "s", basis="引用甲"), [shared]),
                (link("s", "z"), [second]), (link("a", "z", "section_member"), [second])]
    result = assemble([(a, .9), (b, .8), (shared, .7)], loader,
                      options=Options(max_chunks=3, max_supplements=1))
    assert calls == [["a", "b"]]
    assert [c.id for c in result.chunks] == ["a", "b", "s"]
    assert result.roles == ["core", "core", "supplement"]
    assert result.scores == [.9, .8, None]
    assert len(result.reasons["s"]) == 2
    assert "引用甲" in result.context and "引用乙" in result.context and "引用目标" in result.context
    assert "confidence" not in result.context


def test_core_target_dedup_retains_reason_and_unresolved_status():
    Options, assemble = api()
    a, b = chunk("a"), chunk("b", start=10)
    result = assemble([(a, .9), (b, .8)], lambda _: [
        (link("a", "b"), [b]), (link("b", None, status="ambiguous"), [])],
        options=Options(max_chunks=3, max_supplements=1))
    assert len(result.chunks) == 2 and result.roles == ["core", "core"]
    assert result.reasons["b"] and "引用目标" in result.context
    assert any(d.get("status") == "ambiguous" for d in result.decisions)


def test_partial_reference_coverage_and_budget_are_visible():
    Options, assemble = api()
    a, b, c = chunk("a"), chunk("b", start=20), chunk("c", start=40)
    result = assemble([(a, .9)], lambda _: [(link("a", "b", ids=("b", "c")), [b, c])],
                      options=Options(max_chunks=2, max_supplements=1))
    assert [x.id for x in result.chunks] == ["a", "b"]
    assert "部分覆盖" in result.context
    assert any(d.get("status") == "partial_coverage" and d.get("missing_ids") == ["c"] for d in result.decisions)
    assert result.budget_used == len(result.context) <= 6000


def test_overlapping_source_is_rendered_once_and_distinct_ids_count():
    Options, assemble = api()
    a, b, c = chunk("a", "ABCDEF", start=0), chunk("b", "DEFGHI", start=3), chunk("c", "XYZ", start=30)
    result = assemble([(a, .9), (b, .8), (c, .7)], lambda _: [], options=Options(max_chunks=2, max_supplements=0))
    assert len(result.chunks) == 2
    assert "ABCDEFGHI" in result.context and result.context.count("DEF") == 1
    separate = assemble([(a, .9), (c, .7)], lambda _: [], options=Options(max_supplements=0))
    assert "[... omitted ...]" in separate.context


def test_no_semantic_dedup_and_exact_raw_source():
    Options, assemble = api()
    texts = ["2024 年允许 10 人。", "2025 年允许 10 人。", "2025 年不允许 10 人。", "2025 年允许 11 人。"]
    cores = [(chunk(str(i), text, start=i*50), .9-i*.1) for i, text in enumerate(texts)]
    result = assemble(cores, lambda _: [], options=Options(max_supplements=0))
    assert len(result.chunks) == 4 and all(t in result.context for t in texts)


def test_remaining_core_fill_is_not_expanded_and_reserved_budget_fallback():
    Options, assemble = api()
    a, b, c = [chunk(x, x*10, start=i*20) for i, x in enumerate("abc")]
    calls = []
    result = assemble([(a, .9), (b, .8), (c, .7)], lambda ids: calls.append(ids) or [],
                      options=Options(max_chunks=3, max_supplements=1))
    assert calls == [["a", "b"]] and len(result.chunks) == 3
    assert any(d.get("chunk_id") == "c" and d.get("phase") == "unexpanded_core" for d in result.decisions)
    full = assemble([(a, .9)], lambda _: [], options=Options(max_supplements=0))
    calls.clear()
    fallback = assemble([(a, .9)], lambda ids: calls.append(ids) or [],
                        options=Options(max_chunks=2, max_supplements=1, context_budget=full.budget_used))
    assert fallback.status == "ok" and fallback.chunks == [a] and calls == []


def test_long_reason_cannot_escape_supplement_or_total_budget():
    Options, assemble = api()
    a, b = chunk("a"), chunk("b", start=20)
    result = assemble([(a, .9)], lambda _: [(link("a", "b", basis="引用"*500), [b])],
                      options=Options(context_budget=300))
    assert result.chunks == [a] and result.budget_used <= 300
    assert any(d.get("chunk_id") == "b" and d.get("status") == "budget_skipped" for d in result.decisions)


def test_supplement_fraction_charges_separators_and_omission_markers():
    Options, assemble = api()
    a, b = chunk("a", "COREBODY"), chunk("b", "TARGETBODY", start=100)
    loader = lambda _: [(link("a", "b"), [b])]
    baseline = assemble([(a, .9)], loader, options=Options())
    supplement_rendering = baseline.context.split("COREBODY", 1)[1]
    budget = len(supplement_rendering) - 1
    result = assemble([(a, .9)], loader,
                      options=Options(supplement_fraction=budget / 6000))
    assert result.chunks == [a]


def test_selection_exposes_budget_unit_consistently_with_options():
    Options, assemble = api()
    result = assemble([], lambda _: [], options=Options())
    assert result.budget_unit == "characters"


def test_citation_labels_cover_underlying_chunks_and_reason_offsets_are_preserved():
    Options, assemble = api()
    a, b = chunk("a", "ABCDE"), chunk("b", "DEFGH", start=3)
    links = [EvidenceLink("a", "b", "explicit_reference", "附件", source_start=x, source_end=x+2)
             for x in (10, 20)]
    result = assemble([(a, .9)], lambda _: [(item, [b]) for item in links], options=Options())
    assert "[S1]" in result.context and "[S2]" in result.context
    assert [r["source_start"] for r in result.reasons["b"]] == [10, 20]
    assert [r["source_end"] for r in result.reasons["b"]] == [12, 22]


def test_budget_sweep_with_long_annotations_and_partial_references():
    Options, assemble = api()
    a, b, s = chunk("a", "CORE_A"), chunk("b", "CORE_B", start=30), chunk("s", "SUPPORT", start=60)
    # Repeated references have different source coordinates and cannot be
    # deduplicated. Their partial labels can disappear as targets are admitted.
    refs = [EvidenceLink("b", "s", "explicit_reference", "见目标", source_start=i, source_end=i+1)
            for i in range(15)]
    def loader(_):
        return [(link("a", "s"), [s])] + [(ref, [s]) for ref in refs]
    for budget in range(150, 1001, 25):
        result = assemble([(a, .9), (b, .8)], loader,
                          options=Options(max_chunks=3, max_supplements=1, context_budget=budget))
        assert result.budget_used == len(result.context) <= budget
        if "supplement" in result.roles:
            index = result.roles.index("supplement") + 1
            support_header = result.context.index(f"[S{index}]")
            next_header = result.context.find("[S", support_header + 1)
            support_block = result.context[support_header:next_header if next_header >= 0 else None]
            assert len(support_block) <= budget * .3


def test_exact_raw_gap_and_duplicate_ids():
    Options, assemble = api()
    a = chunk("a", "  原文甲\n")
    b = chunk("b", "原文乙  ", start=a.source_end+3, gap="\n \n")
    result = assemble([(a, .9), (a, .8), (b, .7)], lambda _: [], options=Options(max_supplements=0))
    assert [c.id for c in result.chunks] == ["a", "b"]
    assert a.source_text in result.context and b.source_text in result.context
    assert "[... omitted ...]" not in result.context
    assert "\n \n" in result.context


def test_adjacency_is_labeled_and_section_members_are_not_expanded():
    Options, assemble = api()
    a, b, c = chunk("a"), chunk("b", start=20), chunk("c", start=40)
    result = assemble([(a, .9)], lambda _: [
        (link("a", "b", "adjacent"), [b]), (link("a", "c", "section_member"), [c])], options=Options())
    assert [c.id for c in result.chunks] == ["a", "b"]
    assert "邻接语境" in result.context and result.scores == [.9, None]


def test_reject_cross_document_or_unlisted_target():
    Options, assemble = api()
    a, b, c = chunk("a"), chunk("b", doc="other"), chunk("c", start=20)
    result = assemble([(a, .9)], lambda _: [(link("a", "b"), [b, c])], options=Options())
    assert result.chunks == [a]


@pytest.mark.parametrize("score", [None, True, math.nan, math.inf])
def test_core_scores_must_be_finite_numbers(score):
    Options, assemble = api()
    with pytest.raises(ValueError):
        assemble([(chunk("a"), score)], lambda _: [], options=Options())
