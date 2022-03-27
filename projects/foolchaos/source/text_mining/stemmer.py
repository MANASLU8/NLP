import snowballstemmer


def stem(word, lang):
    return snowballstemmer.stemmer(lang=lang).stemWord(word=word)
