import logging
from random import shuffle

from gensim.models import Word2Vec

from dirs import annotated_corpus_dir, word2vec_model_filepath
from task1.token_tag import TokenTag
from task1.utils import read_tokens_from_annotated_corpus_tsv
from task3.token_dictionary import load_token_dictionary

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

_dct = load_token_dictionary()
W2V_VECTOR_SIZE = 100


def _sentences_generator():
    files = list(annotated_corpus_dir.glob("train/*/*"))
    shuffle(files)
    for file in files:
        tokens = read_tokens_from_annotated_corpus_tsv(file)
        tokens["sentence"] = (tokens.tag == TokenTag.PUNCT_SENTENCE).cumsum()
        tokens = tokens[tokens.token.isin(_dct.tokens)]
        sentence_arrays = tokens[["token", "sentence"]].groupby(["sentence"]).agg(lambda v: list(v)).token
        yield from sentence_arrays


# https://jacopofarina.eu/posts/gensim-generator-is-not-iterator/
class SentencesIterator:
    def __init__(self, generator_function):
        self.generator_function = generator_function
        self.generator = self.generator_function()

    def __iter__(self):
        # reset the generator
        self.generator = self.generator_function()
        return self

    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result


if __name__ == '__main__':
    w2v = Word2Vec(SentencesIterator(_sentences_generator), workers=8, epochs=5, vector_size=W2V_VECTOR_SIZE)
    w2v.save(str(word2vec_model_filepath))
