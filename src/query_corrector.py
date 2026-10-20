
import re
from functools import lru_cache

from spellchecker import SpellChecker


# Spelling dictionary for English.
_spell_checker = SpellChecker(language="en")


@lru_cache(maxsize=1)
def _get_grammar_tool():
    """Create the local English grammar checker once."""
    import language_tool_python

    return language_tool_python.LanguageTool("en-US")


def _normalize_whitespace(text: str) -> str:
    """Remove extra spaces, tabs, and line breaks."""
    return re.sub(r"\s+", " ", text).strip()


def _remove_repeated_words(text: str) -> str:
    """Remove accidental consecutive repeated words."""
    return re.sub(
        r"\b([A-Za-z]+)(\s+\1\b)+",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )


def _correct_spelling(text: str) -> str:
    """Correct likely spelling mistakes while protecting names and acronyms."""

    def replace_word(match):
        word = match.group(0)

        # Avoid changing short words, acronyms, and internal capitals.
        if len(word) < 4 or word.isupper():
            return word

        if any(char.isupper() for char in word[1:]):
            return word

        # Preserve title-case words, which may be names.
        if word[0].isupper():
            return word

        correction = _spell_checker.correction(word.lower())

        if correction:
            return correction

        return word

    return re.sub(r"\b[A-Za-z]+\b", replace_word, text)


def correct_query(query: str) -> str:
    """
    Correct a user's English video-search query.

    Processing:
    1. Clean whitespace and repeated words.
    2. Correct likely spelling mistakes.
    3. Correct basic grammar with local LanguageTool.
    4. Clean whitespace again.

    If the grammar engine is unavailable, spelling correction still works.
    """
    if not isinstance(query, str):
        return ""

    original = _normalize_whitespace(query)

    if not original:
        return ""

    corrected = _remove_repeated_words(original)
    corrected = _correct_spelling(corrected)

    try:
        grammar_tool = _get_grammar_tool()
        corrected = grammar_tool.correct(corrected)
    except Exception as exc:
        print(
            "Grammar correction unavailable; "
            f"continuing with spelling correction. Details: {exc}"
        )

    corrected = _remove_repeated_words(corrected)
    corrected = _normalize_whitespace(corrected)

    return corrected or original
