from graphrag.answering import build_answer_messages


CONTAMINATED = ("中华第一库", "Super-N", "Fast-N", "650.33", "60亿元股票回购")


def test_prompt_contains_no_private_fixture_terms():
    rendered = repr(build_answer_messages("问题", "[S1] 资料"))
    assert not any(term in rendered for term in CONTAMINATED)


def test_context_is_passed_with_source_labels():
    messages = build_answer_messages("指标是多少？", "[S1] 指标为 8%。")
    assert messages[0]["role"] == "system"
    assert "[S1]" in messages[1]["content"]
