import os

import nltk
import pandas as pd
from pathlib import Path
from nltk import SnowballStemmer, WordNetLemmatizer
from tokenizer.tokenizer import tokenize_text, classes
from nltk.corpus import wordnet

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

endOfClause = ['.', '?', '!']


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def process_file(fname):
    print('working on ', fname)
    df = pd.read_csv(fname, sep=',', header=None)
    data = df.values
    dataCount = len(data)
    n = 0
    for row in data:
        classId = row[0]
        try:
            dirPath = "../assets/" + Path(fname).name.split('.')[0] + "/" + classes[classId] + '/'
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            f = open(dirPath + str(n) + '.tsv', 'w+')
            f.truncate(0)
            for i in range(1, len(row)):
                text = row[i]
                tokens = tokenize_text(text)

                prev = [0, '', '']
                current_clause = None
                for w in tokens:
                    if w[1] != 'whitespace':
                        f.write(w[2] + '\t' + stemmer.stem(w[2]) + "\t" + lemmatizer.lemmatize(w[2], get_wordnet_pos(w[2])) + '\n')
                    elif prev[2] in endOfClause:
                        f.write('\n')
                    prev = w
                f.write('\n')
            f.close()
        except Exception as e:
            print(e)
            print([n, text, tokens])
            pass
        n = n + 1
        if n % 1000 == 0:
            print(int(n * 100 / dataCount), '%')

def main():
    fname_train = '../assets/raw-dataset/train.csv'
    fname_test = '../assets/raw-dataset/test.csv'
    process_file(fname_train)
    process_file(fname_test)


if __name__ == "__main__":
    main()
