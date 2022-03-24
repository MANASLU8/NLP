from nltk import pos_tag as nltk_pos_tag
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


def lemmatize(word, text):
    lemmer = WordNetLemmatizer()
    tag = wordnet_lemmatize(word=word, text=text)
    return lemmer.lemmatize(word=word, pos=pos_tag(word=word, tag=tag)),


def wordnet_lemmatize(word, text):
    tagged = nltk_pos_tag(word_tokenize(text=text))
    tag = [token for token in tagged if token[0] == word]
    if not tag:
        return None
    return tag[0][1][0].upper()


def pos_tag(word, tag=None):
    if tag is None:
        tag = nltk_pos_tag([word])[0][1][0].upper()
    tags = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tags.get(tag, wordnet.NOUN)
