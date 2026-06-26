"""Multilingual cheap-tier keyword tokenizer (KET-RAG skeleton channel).

The existing cheap-tier tokenizers are ASCII-only: ``importance.py`` matches
``[A-Z][a-z]+`` and ``_accel.extract_keywords`` falls back to
``\\b[a-zA-Z]{3,}\\b``. On Cyrillic / CJK / accented-Latin text those patterns
extract almost nothing, so keyword-based chunk selection degenerates.

``tokenize_multilingual`` uses a Unicode word pattern that matches letters of
*any* script, so it works on Serbian, Russian, Chinese, Japanese, accented
French/German, etc. It is leaf-level (stdlib only) and is consumed only by the
flag-on KET-RAG skeleton path; the ASCII tokenizers keep their behavior.

Stopword handling: only a small English stopword set is stripped (the same set
``_accel.extract_keywords`` uses), so non-English tokens are *never* silently
killed. Words in other languages pass through untouched.

Known limitation: CJK (Han/Kana) text has no whitespace word boundaries, so a
run of CJK characters becomes a single token. This is still a strict
improvement over the ASCII tokenizers (which extract nothing from CJK), but
real CJK segmentation (jieba / MeCab) or character n-grams would give better
cross-chunk keyword overlap. Deferred to a follow-up to avoid pulling a heavy
segmentation dependency into this leaf module.
"""

from __future__ import annotations

import re

from khora._accel import _SKELETON_STOPWORDS

# Letters of ANY script (Unicode), length >= 3. ``[^\W\d_]`` is "word
# character that is not a digit and not underscore", i.e. a letter in any
# script under re.UNICODE.
_MULTILINGUAL_WORD_RE = re.compile(r"[^\W\d_]{3,}", flags=re.UNICODE)


def tokenize_multilingual(text: str) -> list[str]:
    """Extract unique multilingual keywords from text.

    Tokenises with a Unicode letter pattern (any script, length >= 3),
    lowercases, strips a small English stopword set (so non-English tokens are
    never dropped), and deduplicates while preserving first-seen order.

    Args:
        text: Arbitrary text in any language / script.

    Returns:
        Deduplicated lowercase keyword list.
    """
    seen: set[str] = set()
    keywords: list[str] = []
    for match in _MULTILINGUAL_WORD_RE.finditer(text):
        word = match.group().lower()
        if word in _SKELETON_STOPWORDS or word in seen:
            continue
        seen.add(word)
        keywords.append(word)
    return keywords


__all__ = ["tokenize_multilingual"]
