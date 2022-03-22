from pandas import DataFrame

from source.task1.stemmer import add_stems


tokens = DataFrame([
    ("I've", "WORD"),
    ("checked", "WORD"),
    ("what", "WORD"),
    ("that", "WORD"),
    ("function", "WORD"),
    ("does", "WORD"),
    (".", "SENT_END")
], columns=["token", "tag"])


def test_lemmatization():
    add_stems(tokens)
    assert tokens['stem'].to_list() == ["i'v", 'check', 'what', 'that', 'function', 'doe', '.']