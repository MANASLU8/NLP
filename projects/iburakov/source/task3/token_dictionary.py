from functools import cache

import pandas as pd
from nltk.corpus import stopwords
from pandas import DataFrame

from dirs import token_dictionary_filepath

_stopwords = stopwords.words("english")


class TokenDictionary:
    def __init__(self, df: DataFrame):
        self.df = df
        self.tokens = set(df.token)
        self.tokens_list = list(df.token)
        # noinspection PyUnresolvedReferences
        self.counts_by_token = {e.token: e.counts for e in df.itertuples()}


@cache
def load_token_dictionary(filter_stopwords: bool = True) -> TokenDictionary:
    df = pd.read_csv(token_dictionary_filepath, sep="\t", keep_default_na=False)
    if filter_stopwords:
        df = df[~df.token.str.lower().isin(_stopwords)]
        df.reset_index(drop=True, inplace=True)

    return TokenDictionary(df)
