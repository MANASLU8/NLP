import snowballstemmer


def create_stem(word):
    stemmer = snowballstemmer.stemmer('english')
    return stemmer.stemWord(word)
