from pandas import DataFrame
from pytest import mark

from task1.stemmer import get_word_stem, add_stems_to_tokens_dataframe
from task1.token_tag import TokenTag


@mark.parametrize("token,expected_stem", [
    ("easily", "easili"),
    ("users", "user"),
    ("cared", "care"),
    ("university", "univers"),
    ("sing", "sing"),
    ("singing", "sing"),
])
def test_getting_word_stems(token: str, expected_stem: str):
    assert get_word_stem(token) == expected_stem


def test_adding_stems_to_df():
    tokens = DataFrame([
        ("Testing", TokenTag.WORD),
        ("testing", TokenTag.WORD),
        (":", TokenTag.PUNCT_COLON),
        ("3D", TokenTag.WORD),

    ], columns=["token", "tag"])

    add_stems_to_tokens_dataframe(tokens)

    assert tokens.stem.to_list() == ["test", "test", ":", "3d"]
