from nltk import SnowballStemmer
from pandas import DataFrame

from task1.utils import add_mapped_words_to_tokens_dataframe

_stemmer = SnowballStemmer(language="english")


def get_word_stem(word: str) -> str:
    return _stemmer.stem(word)


def add_stems_to_tokens_dataframe(df: DataFrame):
    add_mapped_words_to_tokens_dataframe(df, "stem", get_word_stem)
