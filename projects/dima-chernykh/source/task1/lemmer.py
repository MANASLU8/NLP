import nltk
from nltk import WordNetLemmatizer, pos_tag

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

lemma_handle = WordNetLemmatizer()


def handle_pos(pos):
    pos_switcher = {
        'VERB': 'v',
        'NOUN': 'n',
        'ADJ': 'a',
        'ADV': 'r',
    }
    return pos_switcher.get(pos, "n")


def make_lemma(row):
    if row['tag'] == 'WORD':
        return lemma_handle.lemmatize(row['token'], pos=handle_pos(row['postag'][1]))
    else:
        return row['token']


def add_lemmas(tk_dataframe):
    tokens = tk_dataframe['token'].tolist()
    pos_tags = pos_tag(tokens, tagset='universal')

    tk_dataframe['postag'] = pos_tags
    tk_dataframe['lemma'] = tk_dataframe.apply(lambda row : make_lemma(row), axis=1)
    tk_dataframe.drop('postag', axis=1, inplace=True)
    return tk_dataframe
