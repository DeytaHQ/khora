"""Query normalization for paraphrase-robust retrieval.

Normalizes queries before embedding to improve stability across
paraphrased variants of the same intent.
"""

from __future__ import annotations

import re

# Contractions to expand
_CONTRACTIONS: dict[str, str] = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "can't": "cannot",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "how's": "how is",
    "that's": "that is",
    "there's": "there is",
    "let's": "let us",
}

# Filler words to remove
_FILLERS = frozenset(
    {
        "um",
        "uh",
        "like",
        "basically",
        "actually",
        "literally",
        "honestly",
        "obviously",
        "clearly",
        "well",
        "so",
        "anyway",
        "really",
        "just",
        "very",
        "quite",
        "pretty",
        "rather",
    }
)

# Pre-compile contraction pattern (case-insensitive)
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b",
    re.IGNORECASE,
)


def normalize_query(query: str) -> str:
    """Normalize a query for more robust embedding.

    - Expand contractions (don't → do not)
    - Remove filler words (um, like, basically)
    - Normalize whitespace and punctuation
    - Lowercase
    """
    text = query.lower().strip()

    # Expand contractions
    def _expand(match: re.Match) -> str:
        return _CONTRACTIONS.get(match.group(0).lower(), match.group(0))

    text = _CONTRACTION_RE.sub(_expand, text)

    # Remove filler words (only standalone, preserve as part of larger words)
    words = text.split()
    words = [w for w in words if w not in _FILLERS]
    text = " ".join(words)

    # Normalize excessive punctuation (keep single instances)
    text = re.sub(r"([?!.]){2,}", r"\1", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
