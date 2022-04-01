from typing import Union

import numpy as np
from gensim.models import Word2Vec
from pandas import DataFrame

from paths import word2vec_model_filepath
from task1.token_tag import TokenTag
from task1.tokenizer import tokenize_text
from task3.scripts.train_word2vec import W2V_VECTOR_SIZE
from task3.token_dictionary import load_token_dictionary

_dct = load_token_dictionary()
_w2v = Word2Vec.load(str(word2vec_model_filepath))


def vectorize_text_w2v(text: Union[str, DataFrame]):
    if isinstance(text, str):
        tokens = tokenize_text(text)[["token", "tag"]]
    else:
        tokens = text
    tokens["sentence"] = (tokens.tag == TokenTag.PUNCT_SENTENCE).cumsum()
    tokens = tokens[tokens.token.isin(_dct.tokens)]
    w2vs = DataFrame([_w2v.wv[t] if t in _w2v.wv else np.zeros(W2V_VECTOR_SIZE) for t in tokens.token],
                     index=tokens.token)
    tokens = tokens.set_index("token")
    w2vs["sentence"] = tokens.sentence
    tokens = w2vs
    return tokens.groupby(["sentence"]).mean().mean().values
