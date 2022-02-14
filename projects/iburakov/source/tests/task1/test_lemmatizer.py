from pandas import DataFrame
from pytest import mark

from task1.lemmatizer import get_word_lemma, add_lemmas_to_tokens_dataframe
from task1.token_tag import TokenTag


@mark.parametrize("token,expected_lemma", [
    ("easily", "easily"),
    ("easier", "easy"),
    ("users", "user"),
    ("singing", "sing"),
    ("greatest", "great"),
    ("indices", "index"),
    ("are", "be"),
])
def test_getting_word_lemmas(token: str, expected_lemma: str):
    assert get_word_lemma(token) == expected_lemma


def test_adding_stems_to_df():
    tokens = DataFrame([
        ("The", TokenTag.WORD),
        ("Boy's", TokenTag.WORD),
        ("Cars", TokenTag.WORD),
        ("are", TokenTag.WORD),
        ("Different", TokenTag.WORD),
        ("Colors", TokenTag.WORD),
        ("!", TokenTag.PUNCT_SENTENCE),
    ], columns=["token", "tag"])

    add_lemmas_to_tokens_dataframe(tokens)

    assert tokens.lemma.to_list() == ["the", "boy's", "car", "be", "different", "color", "!"]


def test_lemmatizing_homonyms():
    tokens = DataFrame([
        ("He", TokenTag.WORD),
        ("is", TokenTag.WORD),
        ("a", TokenTag.WORD),
        ("learned", TokenTag.WORD),
        ("man", TokenTag.WORD),
    ], columns=["token", "tag"])

    add_lemmas_to_tokens_dataframe(tokens)

    # "learned" should be an adjective here with the same lemma, actually, but it's mistagged as verb
    assert tokens.lemma.to_list() == ["he", "be", "a", "learn", "man"]

    tokens = DataFrame([
        ("He", TokenTag.WORD),
        ("learned", TokenTag.WORD),
        ("something", TokenTag.WORD),
    ], columns=["token", "tag"])

    add_lemmas_to_tokens_dataframe(tokens)

    assert tokens.lemma.to_list() == ["he", "learn", "something"]
