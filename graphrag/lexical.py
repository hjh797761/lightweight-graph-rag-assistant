"""Shared deterministic Chinese/English terms for SQLite FTS5.

Store ``fts_index_text(raw_text)`` in an FTS5 column using ``unicode61``
(SQLite's default), and pass ``fts_query(raw_query)`` as a bound MATCH value.
Do not apply a stemming tokenizer. Terms use ``t`` followed by UTF-8 hex:
this reversible encoding preserves underscores/hyphens as one FTS token and
cannot collide with another normalized term. Original source text belongs in
a separate column. Empty queries return ``''``; callers should skip MATCH.
"""

import re
import unicodedata


# CJK Unified Ideographs, Extension A, supplementary extensions, and
# compatibility ideographs. Punctuation and ASCII identifiers delimit runs.
_HAN = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\U00030000-\U000323af"
_RUNS = re.compile(rf"[{_HAN}]+|[a-z0-9_-]+")


def lexical_terms(text: str) -> list[str]:
    """NFKC/lowercase identifiers and overlapping Chinese-run bigrams.

    Single-character Chinese runs retain that character. Occurrences remain
    in source order, including duplicates, to retain BM25 term frequencies.
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    terms: list[str] = []
    for match in _RUNS.finditer(normalized):
        run = match.group()
        if run[0].isascii():
            if any(character.isalnum() for character in run):
                terms.append(run)
        elif len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return terms


def _encoded_terms(text: str) -> list[str]:
    return ["t" + term.encode("utf-8").hex() for term in lexical_terms(text)]


def fts_index_text(text: str) -> str:
    """Return the token representation to persist in the FTS5 column."""
    return " ".join(_encoded_terms(text))


def fts_query(text: str) -> str:
    """Return quoted OR literal terms; user input cannot supply FTS syntax."""
    return " OR ".join(
        '"' + term.replace('"', '""') + '"'
        for term in dict.fromkeys(_encoded_terms(text))
    )
