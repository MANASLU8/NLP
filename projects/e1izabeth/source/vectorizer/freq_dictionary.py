import os
from os import listdir

import pandas as pd

from utils import eng_stopwords


dirnames_annotated = ['../assets/train/Business', '../assets/train/World', '../assets/train/Sports',
                      '../assets/train/Sci-Tech']


def construct_entry_info(tokens_entry):
    for dir in dirnames_annotated:
        print('processing ', dir)
        ff = listdir(dir)
        n = 0
        for file in ff:
            n = n + 1
            if int(n % 100) == 0:
                print('\t', n, '/', len(ff))

            filename = dir + '/' + file
            try:
                df = pd.read_csv(filename, delimiter="\t", header=None, usecols=[0, 1], engine='python', error_bad_lines=False)
                for index, item in df.iterrows():
                    word = str(item[1].lower()).strip()
                    if (word not in eng_stopwords) and (item[0] == 'word'):
                        token = word
                        tentry = tokens_entry.get(token)
                        if tentry is None:
                            tentry = dict()
                            tokens_entry[token] = tentry
                        tfentry = tentry.get(filename)
                        if tfentry is None:
                            tfentry = 0
                        tentry[filename] = tfentry + 1
            except Exception as e:
                print('error while reading  ', filename)
                print(str(e))
                pass


def reconstruct_entry_info(old_tokens_entry, dirnames):
    tokens_entry = dict()
    for word in old_tokens_entry:
        tokens_entry[word] = dict()

    for dir in dirnames:
        print('processing ', dir)
        ff = listdir(dir)
        n = 0
        for file in ff:
            n = n + 1
            if int(n % 100) == 0:
                print('\t', n, '/', len(ff))

            filename = dir + '/' + file
            try:
                df = pd.read_csv(filename, delimiter="\t", header=None, usecols=[0, 1], engine='python', error_bad_lines=False)
                for index, item in df.iterrows():
                    word = str(item[1].lower()).strip()
                    if (word not in eng_stopwords) and (item[0] == 'word'):
                        token = word
                        tentry = tokens_entry.get(token)
                        if not tentry is None:
                            tfentry = tentry.get(filename)
                            if tfentry is None:
                                tfentry = 0
                            tentry[filename] = tfentry + 1
            except Exception as e:
                print('error while reading  ', filename)
                print(str(e))
                pass
    return tokens_entry


def write_tokens_freq_dictionary(tokens_entry, vec_dict_file_path):
    with open(vec_dict_file_path, "w") as vec_dict_file:
        for word, tentry in tokens_entry.items():
            vec_dict_file.write(word)
            vec_dict_file.write('\n')
            for fname, count in tentry.items():
                vec_dict_file.write(':')
                vec_dict_file.write(fname)
                vec_dict_file.write('\n')
                vec_dict_file.write('-')
                vec_dict_file.write(str(count))
                vec_dict_file.write('\n')
            vec_dict_file.flush()


def read_freq_dict(fname):
    sz = os.path.getsize(fname)
    tokens_entry = dict()
    tentry = None
    filename = None
    with open(fname, "r") as f:
        line = f.readline()
        n = 0
        while line:
            n = n + 1
            if n % 10000 == 0:
                print('\t', f.tell() / sz * 100, '%')
            token = line.strip()

            if token.startswith(":"):
                filename = token[1:]
            elif token.startswith("-"):
                tentry[filename] = int(token[1:])
            else:
                tentry = tokens_entry.get(token)
                if tentry is None:
                    tentry = dict()
                    tokens_entry[token] = tentry

            line = f.readline()

    return tokens_entry
