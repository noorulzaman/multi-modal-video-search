import pytest

from src import query_corrector
from src.query_corrector import correct_query


class PassThroughGrammarTool:
    def correct(self, text):
        return text


@pytest.fixture(autouse=True)
def no_language_tool(monkeypatch):
    """Keep tests offline and free of the Java-based LanguageTool server."""
    monkeypatch.setattr(
        query_corrector, "_get_grammar_tool", lambda: PassThroughGrammarTool()
    )


@pytest.mark.parametrize("value", [None, 42, "", "   \n\t "])
def test_empty_or_invalid_input_returns_empty_string(value):
    assert correct_query(value) == ""


def test_whitespace_is_normalized():
    assert correct_query("  a   red \n car ") == "a red car"


def test_repeated_words_are_removed():
    assert correct_query("a dog dog running") == "a dog running"


def test_spelling_is_corrected():
    assert correct_query("a person walkng") == "a person walking"


def test_acronyms_names_and_short_words_are_preserved():
    query = "NASA rocket near Paris iPhone"

    assert correct_query(query) == query


def test_grammar_failure_falls_back_to_spelling(monkeypatch):
    def broken_tool():
        raise RuntimeError("Java is not installed")

    monkeypatch.setattr(query_corrector, "_get_grammar_tool", broken_tool)

    assert correct_query("a person walkng") == "a person walking"
