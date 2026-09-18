import importlib
import sqlite3

import pytest


def lexical():
    return importlib.import_module("graphrag.lexical")


def test_normalizes_and_preserves_identifier_terms_and_occurrences():
    assert lexical().lexical_terms("ＡＢ_１２ Release-V2 ab_12 123") == [
        "ab_12", "release-v2", "ab_12", "123"
    ]


def test_chinese_bigrams_do_not_cross_punctuation_or_identifier_boundaries():
    assert lexical().lexical_terms("北京，大学 AB_12") == ["北京", "大学", "ab_12"]
    assert lexical().lexical_terms("北京大学，京A学") == ["北京", "京大", "大学", "京", "a", "学"]


def test_empty_or_only_punctuation_has_no_query():
    assert lexical().lexical_terms(' ,。()" ') == []
    assert lexical().fts_index_text("") == ""
    assert lexical().fts_query(' ,。()" ') == ""


@pytest.fixture
def index():
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE VIRTUAL TABLE documents USING fts5(body, tokenize='unicode61')")
        yield connection


def add(index, texts):
    index.executemany("INSERT INTO documents(body) VALUES (?)", [(lexical().fts_index_text(text),) for text in texts])


def matches(index, query):
    return index.execute(
        "SELECT rowid FROM documents WHERE documents MATCH ? ORDER BY rowid",
        (lexical().fts_query(query),),
    ).fetchall()


def test_fts_preserves_exact_identifier_equality(index):
    add(index, ["AB_12", "AB 12", "AB-12", "ab_12x", "t61625f3132"])
    assert matches(index, "ab_12") == [(1,)]
    assert matches(index, "AB-12") == [(3,)]
    assert matches(index, "t61625f3132") == [(5,)]


def test_fts_user_operators_quotes_parentheses_are_literal_terms(index):
    add(index, ["OR", "NEAR", "needle", "unrelated"])
    assert matches(index, 'OR NEAR("needle")') == [(1,), (2,), (3,)]
    query = lexical().fts_query('OR NEAR("needle")')
    assert len(query.split(" OR ")) == 3
    assert all(term.startswith('"') and term.endswith('"') for term in query.split(" OR "))


def test_chinese_matching_uses_shared_representation(index):
    add(index, ["北京，大学", "京大", "北京大学"])
    assert matches(index, "北京") == [(1,), (3,)]
    assert matches(index, "京大") == [(2,), (3,)]


def test_actual_fts_bm25_rewards_repeated_matching_terms(index):
    add(index, ["ab_12 filler filler", "ab_12 ab_12 filler", "unrelated filler filler"])
    rows = index.execute(
        "SELECT rowid, bm25(documents) FROM documents WHERE documents MATCH ? ORDER BY bm25(documents)",
        (lexical().fts_query("ab_12"),),
    ).fetchall()
    assert [row[0] for row in rows] == [2, 1]
    assert rows[0][1] < rows[1][1] < 0
