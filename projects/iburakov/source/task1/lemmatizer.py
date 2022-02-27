from nltk import WordNetLemmatizer, pos_tag, download
from nltk.corpus import wordnet
from pandas import DataFrame

from task1.utils import add_mapped_words_to_tokens_dataframe

download("wordnet", quiet=True)
download("omw-1.4", quiet=True)
download("averaged_perceptron_tagger", quiet=True)

_lemmatizer = WordNetLemmatizer()
_wordnet_pos_map = {
    "N": wordnet.NOUN,
    "J": wordnet.ADJ,
    "V": wordnet.VERB,
    "R": wordnet.ADV,
}


def get_word_lemma(word: str) -> str:
    # possible improvement: context-dependent tagging
    word = word.lower()
    pos_suggestions = pos_tag([word])
    pos = _wordnet_pos_map.get(pos_suggestions[0][1][0], None)
    if pos is not None:
        return _lemmatizer.lemmatize(word, pos)
    else:
        return word


def add_lemmas_to_tokens_dataframe(df: DataFrame):
    add_mapped_words_to_tokens_dataframe(df, "lemma", get_word_lemma)
