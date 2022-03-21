from pandas import DataFrame

from source.task1.lemmer import add_lemmas



def test_lemmatization():
    tokens = DataFrame([
        ("Hello", "WORD"),
        ("I'm", "WORD"),
        ("looking", "WORD"),
        ("out", "WORD"),
        ("for", "WORD"),
        ("my", "WORD"),
        ("friends", "WORD"),
        (".", "SENT_END")
    ], columns=["token", "tag"])
    add_lemmas(tokens)
    assert tokens['lemma'].to_list() == ['Hello', "I'm", 'look', 'out', 'for', 'my', 'friend', '.']
    tokens = DataFrame([
        ("Please", "WORD"),
        ("e-mail", "WORD"),
        (",", "PUNC_COMMA"),
        ("as", "WORD"),
        ("I", "WORD"),
        ("don't", "WORD"),
        ("read", "WORD"),
        ("this", "WORD"),
        ("group", "WORD"),
        ("very", "WORD"),
        ("often", "WORD"),
        (".", "SENT_END")
    ], columns=["token", "tag"])
    add_lemmas(tokens)
    assert tokens['lemma'].to_list() == ['Please', "e-mail", ',', 'a', 'I', "don't", 'read', 'this', 'group', 'very', 'often', '.']

