from sklearn.feature_extraction.text import TfidfTransformer

from task1.token_tag import TokenTag
from task1.tokenizer import tokenize_text
from task3.term_document_matrix import load_term_document_matrix
from task3.token_dictionary import load_token_dictionary

_dct = load_token_dictionary()
_tdm = load_term_document_matrix()
_tfidf = TfidfTransformer()
_tfidf.fit(_tdm)


def vectorize_text_simple(text: str):
    tokens = tokenize_text(text)[["token", "tag"]]
    tokens["sentence"] = (tokens.tag == TokenTag.PUNCT_SENTENCE).cumsum()
    tokens = tokens[tokens.token.isin(_dct.tokens)]
    tokens = tokens.groupby(["sentence", "token"]).count()
    tokens = tokens.unstack(level="token", fill_value=0.0)
    tokens = tokens.droplevel(0, axis=1)
    tokens = tokens.reindex(_dct.tokens_list, axis=1, fill_value=0.0)
    return _tfidf.transform(tokens.values).mean(axis=0).A1
