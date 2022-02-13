import os
from pathlib import Path

import pandas as pd
from nltk import SnowballStemmer, WordNetLemmatizer

from tokenizer.tokenizer import tokenize_text, classes

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

def process_file(fname):
    print('working on ', fname)
    df = pd.read_csv(fname, sep=',', header=None)
    data = df.values
    dataCount = len(data)
    n = 0
    for row in data:
        classId = row[0]
        text = row[1]
        tokens = tokenize_text(text)
        try:
            dirPath = "../assets/" + Path(fname).name.split('.')[0] + "/" + classes[classId] + '/'
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            f = open(dirPath + str(n) + '.tsv', 'w+')
            f.truncate(0)
            for w in tokens:
                f.write(w[2] + '\t' + stemmer.stem(w[2]) + "\t" + lemmatizer.lemmatize(w[2]) + '\n')
            f.close()
        except Exception as e:
            print(e)
            print([n, text, tokens])
            pass
        n = n + 1
        if n%1000==0:
            print(int(n*100/dataCount), '%')

def main():
    print(lemmatizer.lemmatize("companies"))
    fname_train = '../assets/raw-dataset/train.csv'
    fname_test = '../assets/raw-dataset/test.csv'
    process_file(fname_train)
    process_file(fname_test)

if __name__ == "__main__":
    main()
