from nltk import SnowballStemmer

snowball = SnowballStemmer("english")


def stem(row):
    if row['tag'] == 'WORD':
        return snowball.stem(row['token'])
    else:
        return row['token']


def add_stems(tk_dataframe):
    tk_dataframe['stem'] = tk_dataframe.apply(lambda row: stem(row), axis=1)
    return tk_dataframe
