"""Unit tests for the multilingual cheap-tier tokenizer (KET-RAG skeleton).

Proves ``tokenize_multilingual`` extracts keywords from Cyrillic, CJK, and
accented-Latin text where the existing ASCII-only tokenizers return ~nothing.
"""

from __future__ import annotations

import re
import unicodedata

from khora._accel import _KEYWORD_RE
from khora.extraction.importance import _CAPITALIZED_PHRASE_RE, _PROPER_NOUN_RE
from khora.extraction.tokenize import is_cjk_token, tokenize_core, tokenize_multilingual


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


def test_cjk_bigrams() -> None:
    """CJK text yields non-degenerate overlapping bigrams (#1388)."""
    chinese = tokenize_multilingual("玛丽居里发现了镭元素")
    assert chinese  # non-empty
    # Bigrams (not one collapsed run): "居里" is its own token now.
    assert "居里" in chinese
    assert all(len(tok) == 2 for tok in chinese)
    # Query/index consistency: a sub-phrase's bigrams are a subset.
    sub = tokenize_multilingual("居里发现")
    assert sub
    assert set(sub) <= set(chinese)


def test_indic_devanagari_kept_whole() -> None:
    """Devanagari base + matras stay one non-empty token (#1388), not dropped."""
    tokens = tokenize_multilingual("हिन्दी")
    assert tokens == ["हिन्दी"]


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


def test_core_ascii_byte_identical_to_regex() -> None:
    """tokenize_core is byte-identical to ``[^\\W_]+`` on pure-ASCII input (#1388)."""
    text = "Marie Curie discovered radium element 1898"
    expected = re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text.lower()))
    assert tokenize_core(text) == expected
    assert tokenize_core(text) == ["marie", "curie", "discovered", "radium", "element", "1898"]


def test_core_underscore_is_a_boundary() -> None:
    """Underscore is excluded (a boundary), matching the old ``[^\\W_]+`` (#1388)."""
    assert tokenize_core("foo_bar") == ["foo", "bar"]


def test_core_cjk_bigrams_and_unigram() -> None:
    """CJK runs bigram; a single CJK char is a unigram (#1388)."""
    assert tokenize_core("玛丽居")[:2] == ["玛丽", "丽居"]
    assert tokenize_core("镭") == ["镭"]


def test_core_indic_keeps_marks() -> None:
    """Category-M marks are word-forming so Devanagari stays one token (#1388)."""
    assert tokenize_core("हिन्दी") == ["हिन्दी"]
    # The regex the old path used fragments it into single base consonants.
    assert re.findall(r"[^\W_]+", "हिन्दी") == ["ह", "न", "द"]


def test_core_mixed_script_run_segments_at_boundary() -> None:
    """A run mixing CJK and non-CJK splits at the script boundary (#1388).

    Without segmentation the whole run bigrams across scripts, producing
    cross-script garbage like "n语" and dropping the clean Latin/numeric token.
    """
    assert tokenize_core("Python语言") == ["python", "语言"]
    assert tokenize_core("東京2024") == ["東京", "2024"]


def test_is_cjk_token() -> None:
    """is_cjk_token flags CJK bigrams/unigrams but not Latin/Cyrillic (#1388)."""
    assert is_cjk_token("玛丽")
    assert is_cjk_token("镭")
    assert is_cjk_token("日本")  # kanji
    assert is_cjk_token("テキ")  # katakana
    assert not is_cjk_token("radium")
    assert not is_cjk_token("मार")  # Devanagari is not CJK
    assert not is_cjk_token("кири")
