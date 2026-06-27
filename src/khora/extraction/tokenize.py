"""Multilingual, dependency-free keyword tokenizer (shared core).

The original cheap-tier tokenizers were ASCII-only: ``importance.py`` matched
``[A-Z][a-z]+`` and ``_accel.extract_keywords`` fell back to
``\\b[a-zA-Z]{3,}\\b``. On Cyrillic / CJK / accented-Latin text those patterns
extracted almost nothing, so keyword-based chunk selection degenerated. #1383
(``tokenize_multilingual``) and #1384 (BM25 ``query/keyword.py``) fixed the
*space-separated, base-letter* scripts (Cyrillic, Greek, (N)FC Latin) by
switching to a Unicode word pattern, but left two gaps that a regex tweak
cannot close:

* **CJK** (Han / Kana / Hangul): no inter-word spaces, so ``[^\\W_]+`` matches
  an entire run as ONE token. It survives, but gives no sub-phrase matching and
  no cross-chunk keyword overlap.
* **Combining-mark scripts** (Indic: Devanagari, Tamil, Bengali, ...): ``\\w``
  does not match the nonspacing/spacing marks (matras), so a word splits at
  every mark into single base consonants, which a ``len > 2`` filter then drops
  entirely -> effectively zero recall.

``tokenize_core`` fixes both, dependency-free (no jieba / MeCab / regex module),
and is shared between BM25 (``query/keyword.py``) and the cheap-tier
``tokenize_multilingual`` below. Algorithm:

1. NFC-normalize + lowercase (keeps #1384's NFC behavior, so a composed query
   matches a decomposed document).
2. Group maximal runs of *word-forming* characters, where word-forming means
   ``unicodedata.category(ch)[0] in {"L", "N", "M"}`` and ``ch != "_"``.
   Including category M (marks) keeps an Indic base consonant and its matras as
   ONE run (fixes Indic fragmentation). This is a Python scan, not a regex,
   because stdlib ``re`` cannot match ``\\p{M}``.
3. For each run, if it is CJK (contains a Han / Hiragana / Katakana /
   Hangul-syllable codepoint), emit overlapping character **bigrams** (e.g.
   "玛丽居" -> "玛丽", "丽居") so CJK gets sub-token matching with index/query
   consistency; a single-char CJK run emits that char as a unigram. Non-CJK
   runs are emitted as the whole run (one token), exactly as the #1384 pattern
   did.

On pure-ASCII input ``tokenize_core`` yields byte-identical tokens to
``[^\\W_]+``: ASCII letters/digits are category L/N, there are no marks, and
nothing is CJK, so every run is emitted whole. English ranking is unchanged.
"""

from __future__ import annotations

import unicodedata

from khora._accel import _SKELETON_STOPWORDS

# CJK codepoint ranges (inclusive). Han ideographs, kana, and Hangul syllables
# have no intra-word spaces, so we segment them with character bigrams instead
# of emitting a whole run as one token.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x3041, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0xAC00, 0xD7A3),  # Hangul Syllables
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2EBEF),  # CJK Unified Ideographs Extensions C-F
)


def _is_cjk(ch: str) -> bool:
    """True if ``ch`` is a Han / kana / Hangul-syllable character."""
    cp = ord(ch)
    for lo, hi in _CJK_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def _is_word_char(ch: str) -> bool:
    """True if ``ch`` is word-forming: a letter, number, or mark (not ``_``)."""
    if ch == "_":
        return False
    return unicodedata.category(ch)[0] in ("L", "N", "M")


def tokenize_core(text: str) -> list[str]:
    """Split text into tokens, NFC-normalized and lowercased.

    Maximal runs of word-forming characters (letters, numbers, marks; not
    underscore) are emitted as whole tokens, except CJK runs, which are emitted
    as overlapping character bigrams (single-char CJK runs emit a unigram).
    Byte-identical to ``[^\\W_]+`` on pure-ASCII input.

    Args:
        text: Arbitrary text in any language / script.

    Returns:
        List of tokens in order (with duplicates; callers dedupe if needed).
    """
    normalized = unicodedata.normalize("NFC", text.lower())
    tokens: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if not run:
            return
        if any(_is_cjk(c) for c in run):
            if len(run) == 1:
                tokens.append(run[0])
            else:
                for i in range(len(run) - 1):
                    tokens.append(run[i] + run[i + 1])
        else:
            tokens.append("".join(run))
        run.clear()

    for ch in normalized:
        if _is_word_char(ch):
            run.append(ch)
        else:
            flush()
    flush()

    return tokens


def is_cjk_token(token: str) -> bool:
    """True if ``token`` contains any CJK character.

    Used by callers to exempt CJK n-gram tokens from Latin-oriented short-token
    length filters (CJK bigrams are length 2 and must survive).
    """
    return any(_is_cjk(c) for c in token)


def tokenize_multilingual(text: str) -> list[str]:
    """Extract unique multilingual keywords from text (cheap-tier KET-RAG).

    Routes through :func:`tokenize_core` (any script, mark-aware, CJK bigrams),
    strips a small English stopword set (so non-English tokens are never
    dropped), drops short Latin/ASCII tokens (length < 3) while keeping CJK
    n-gram tokens, and deduplicates while preserving first-seen order.

    Args:
        text: Arbitrary text in any language / script.

    Returns:
        Deduplicated lowercase keyword list.
    """
    seen: set[str] = set()
    keywords: list[str] = []
    for word in tokenize_core(text):
        if word in _SKELETON_STOPWORDS or word in seen:
            continue
        # Keep CJK n-gram tokens regardless of length; only short non-CJK
        # tokens are dropped (matches the prior length >= 3 behavior on Latin).
        if len(word) < 3 and not is_cjk_token(word):
            continue
        seen.add(word)
        keywords.append(word)
    return keywords


__all__ = ["tokenize_core", "is_cjk_token", "tokenize_multilingual"]
