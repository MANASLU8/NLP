import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


def get_wordnet_pos(word, tag):
    """Map POS tag to first character lemmatize() accepts"""
    if tag is None:
        tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def wordnet_lemmatize_with_sentence(word, sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    word_tag = [nltk_token for nltk_token in nltk_tagged if nltk_token[0] == word]
    if not word_tag:
        return None
    return word_tag[0][1][0].upper()


def wordnet_lemmatize(word, initial_text):
    lemmatizer = WordNetLemmatizer()
    word_tag = wordnet_lemmatize_with_sentence(word, initial_text)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(word, word_tag)), get_wordnet_pos(word, word_tag)]
