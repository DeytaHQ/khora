"""Unit tests for the multilingual cheap-tier tokenizer (KET-RAG skeleton).

Proves ``tokenize_multilingual`` extracts keywords from Cyrillic, CJK, and
accented-Latin text where the existing ASCII-only tokenizers return ~nothing.
"""

from __future__ import annotations

from khora._accel import _KEYWORD_RE
from khora.extraction.importance import _CAPITALIZED_PHRASE_RE, _PROPER_NOUN_RE
from khora.extraction.tokenize import tokenize_multilingual


def test_serbian_latin_cyrillic() -> None:
    """Serbian text (Latin and Cyrillic) tokenizes to real keywords."""
    latin = tokenize_multilingual("Marija Kiri je otkrila radijum")
    # "je" is dropped (length < 3); the rest survive, lowercased.
    assert latin == ["marija", "kiri", "otkrila", "radijum"]

    cyrillic = tokenize_multilingual("Марија Кири је открила радијум")
    assert "марија" in cyrillic
    assert "открила" in cyrillic
    assert "радијум" in cyrillic


def test_russian_cyrillic() -> None:
    russian = tokenize_multilingual("Мария Кюри открыла радий")
    assert "мария" in russian
    assert "открыла" in russian
    assert "радий" in russian


def test_cjk() -> None:
    """CJK text yields at least one keyword (vs nothing from ASCII regexes)."""
    chinese = tokenize_multilingual("玛丽居里发现了镭元素")
    assert chinese  # non-empty
    assert any("镭" in tok or "居里" in tok for tok in chinese)


def test_accented_latin() -> None:
    accented = tokenize_multilingual("café résumé naïve Zürich")
    assert "café" in accented
    assert "résumé" in accented
    assert "naïve" in accented
    assert "zürich" in accented


def test_dedup_and_lowercase() -> None:
    tokens = tokenize_multilingual("Apple apple APPLE banana Banana")
    assert tokens == ["apple", "banana"]


def test_english_stopwords_stripped_non_english_kept() -> None:
    """English stopwords are stripped; non-English tokens are never dropped."""
    # "the" and "and" are English stopwords; "otkrila"/"radijum" are not.
    tokens = tokenize_multilingual("the radijum and otkrila")
    assert "the" not in tokens
    assert "and" not in tokens
    assert "radijum" in tokens
    assert "otkrila" in tokens


def test_ascii_regexes_return_nothing_on_non_latin() -> None:
    """The old ASCII-only tokenizers extract ~nothing on Cyrillic/CJK."""
    cyrillic = "Марија Кири открила радијум"
    cjk = "玛丽居里发现了镭元素"

    assert _CAPITALIZED_PHRASE_RE.findall(cyrillic) == []
    assert _PROPER_NOUN_RE.findall(cyrillic) == []
    assert _KEYWORD_RE.findall(cyrillic) == []

    assert _CAPITALIZED_PHRASE_RE.findall(cjk) == []
    assert _PROPER_NOUN_RE.findall(cjk) == []
    assert _KEYWORD_RE.findall(cjk) == []

    # ...while the multilingual tokenizer extracts real keywords from both.
    assert tokenize_multilingual(cyrillic)
    assert tokenize_multilingual(cjk)
