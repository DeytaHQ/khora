"""Coverage for ``khora.query.normalization``.

The module is a small pure-function pipeline (contractions, fillers,
punctuation, whitespace, lowercase). Covering each transformation
independently keeps regressions visible — historically all 16 statements
were uncovered.
"""

from __future__ import annotations

import pytest

from khora.query.normalization import normalize_query


@pytest.mark.unit
class TestNormalizeQueryContractions:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Don't go", "do not go"),
            ("Won't break", "will not break"),
            ("I'm here", "i am here"),
            ("You're right", "you are right"),
            ("It's late", "it is late"),
            ("That's wrong", "that is wrong"),
            ("Where's the file?", "where is the file?"),
            ("Let's start", "let us start"),
            # Case-insensitive — caller may type any case.
            ("CAN'T do it", "cannot do it"),
        ],
    )
    def test_expands_known_contractions(self, raw: str, expected: str) -> None:
        assert normalize_query(raw) == expected

    def test_leaves_unknown_contractions_unchanged(self) -> None:
        # "y'all" is not in the contractions map; expansion must leave
        # the unknown form alone rather than dropping the apostrophe.
        out = normalize_query("y'all stay")
        assert "y'all" in out


@pytest.mark.unit
class TestNormalizeQueryFillers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("um what is this", "what is this"),
            ("really very quite good", "good"),
            ("well so basically test", "test"),
        ],
    )
    def test_removes_standalone_filler_words(self, raw: str, expected: str) -> None:
        assert normalize_query(raw) == expected

    def test_punctuation_attached_to_filler_blocks_removal(self) -> None:
        # split() preserves punctuation as part of the token, so 'like,'
        # is not equal to 'like' and stays in the output. This is an
        # intentional limitation of the cheap split-based approach —
        # documenting it here so any future "smarter" filler removal is a
        # deliberate decision, not a silent behaviour change.
        assert "like," in normalize_query("like, that is fine")

    def test_preserves_fillers_inside_other_words(self) -> None:
        # 'just' is a filler, but 'justice' isn't — split() bounds it,
        # the filter only drops whole-word matches.
        assert "justice" in normalize_query("Justice is just")

    def test_keeps_query_intact_when_no_fillers_present(self) -> None:
        assert normalize_query("Postgres replication lag") == "postgres replication lag"


@pytest.mark.unit
class TestNormalizeQueryPunctuationAndWhitespace:
    def test_collapses_repeated_question_marks(self) -> None:
        assert normalize_query("Really??") == "really?"

    def test_collapses_repeated_exclamation_marks(self) -> None:
        assert normalize_query("Wow!!!") == "wow!"

    def test_collapses_repeated_periods(self) -> None:
        # Wait... two-or-more periods collapse to one. Single periods
        # are unaffected.
        assert normalize_query("Hmm...") == "hmm."

    def test_single_punctuation_unchanged(self) -> None:
        assert normalize_query("What?") == "what?"

    def test_collapses_runs_of_whitespace(self) -> None:
        assert normalize_query("foo   bar\tbaz\n\nqux") == "foo bar baz qux"

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        assert normalize_query("   hello world   ") == "hello world"


@pytest.mark.unit
class TestNormalizeQueryComposition:
    def test_all_transforms_apply_together(self) -> None:
        # Tokens picked so each transform is exercised independently:
        # - "I'm" → contraction expansion
        # - "um" / "really" → standalone fillers (clean tokens — no
        #   trailing punctuation, which would block removal)
        # - "!!" → repeated-punctuation collapse
        # - extra whitespace + leading/trailing spaces
        out = normalize_query("  Um I'm really tired!!  ")
        assert out == "i am tired!"

    def test_empty_query_returns_empty(self) -> None:
        assert normalize_query("") == ""

    def test_whitespace_only_query_returns_empty(self) -> None:
        assert normalize_query("   \t\n  ") == ""

    def test_idempotent_on_normalized_input(self) -> None:
        once = normalize_query("don't worry um, it's fine!!")
        twice = normalize_query(once)
        assert once == twice
