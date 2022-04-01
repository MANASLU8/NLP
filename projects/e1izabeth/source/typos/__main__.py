import os
from os import listdir
from pathlib import Path

import pandas as pd
import regex

from tokenizer.tokenizer import tokenize_text

classes = dict([
    (1, 'World'),
    (2, 'Sports'),
    (3, 'Business'),
    (4, 'Sci-Tech')
])

DICT_NAME = 'dictionary'


def create_dictionary(dirnames_annotated):
    with open('../assets/' + DICT_NAME, "w") as dictionary_file:
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
                    df = pd.read_csv(filename, delimiter="\t", header=None, usecols=[0], engine='python',
                                     error_bad_lines=False)
                    for item in df.iloc[:, 1].tolist():
                        dictionary_file.write(str(item).strip())
                        dictionary_file.write('\n')
                except Exception as e:
                    print('error while reading  ', filename)
                    print(str(e))
                    pass


def process_file(fname):
    df = pd.read_csv(fname, sep=',', header=None)
    data = df.values
    data_count = len(data)
    n = 0
    for row in data:
        class_id = row[0]
        try:
            dir_path = "../assets/" + Path(fname).name.split('.')[0] + "/" + classes[class_id] + '/'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            f = open(dir_path + str(n) + '.tsv', 'w')
            f.truncate(0)
            for i in range(1, len(row)):
                text = row[i]
                tokens = tokenize_text(text)
                for w in tokens:
                    if w[1] == 'word':
                        f.write(w[2])
                    else:
                        f.write(w[2])
            f.close()
        except Exception as e:
            print(e)
            print([n, text, tokens])
            pass
        n = n + 1
        if n % 1000 == 0:
            print(int(n * 100 / data_count), '%')


def score(source_file, corrupted_file, fixed_file):
    total_count = 0
    good_tokens_count_before = 0
    good_tokens_count_after = 0
    sf = pd.read_csv(source_file, sep=',', header=None)
    cf = pd.read_csv(corrupted_file, sep=',', header=None)
    ff = pd.read_csv(fixed_file, sep=',', header=None)
    rowsCount = len(ff.values)
    for i in range(0, rowsCount):
        source_row = sf.values[i]
        dirty_row = cf.values[i]
        fixed_row = ff.values[i]
        for j in range(1, len(source_row)):
            source_tokens = tokenize_text(source_row[j])
            dirty_tokens = tokenize_text(dirty_row[j])
            fixed_tokens = tokenize_text(fixed_row[j])
            if len(source_tokens) == len(dirty_tokens) and len(source_tokens) == len(fixed_tokens):
                for k in range(0, len(source_tokens)):
                    if source_tokens[k][1] == 'word' and dirty_tokens[k][1] == 'word' and fixed_tokens[k][1] == 'word' and regex.match('^[a-zA-Z]+$', source_tokens[k][2]):
                        total_count = total_count + 1
                        if source_tokens[k][2].lower() == dirty_tokens[k][2].lower():
                            good_tokens_count_before = good_tokens_count_before + 1
                        if source_tokens[k][2].lower() == fixed_tokens[k][2].lower():
                            good_tokens_count_after = good_tokens_count_after + 1
        print('\t', i, '/', rowsCount)
    print('total_count: ', total_count)
    print('good_tokens_count_before: ', good_tokens_count_before)
    print('good_tokens_count_after: ', good_tokens_count_after)
    print('result: ', (good_tokens_count_after / total_count) - (good_tokens_count_before / total_count))
    errors_before = total_count - good_tokens_count_before
    errors_after = total_count - good_tokens_count_after
    print((errors_before-errors_after)*100/errors_before, '% error fixed correctly')


def main():
    fname_corrupted = '../assets/raw-dataset/test-corrupted.csv'
    fname_source = '../assets/raw-dataset/test.csv'
    fname_fixed = '../assets/raw-dataset/test-corrupted.csv.h.out' #test-corrupted.csv.td.out
    fname_dictionary = '../assets/dictionary-no-dedup' # '../assets/' + DICT_NAME
    dirnames_annotated = ['../assets/train/Business', '../assets/train/World', '../assets/train/Sports',
                          '../assets/train/Sci-Tech']

    create_dictionary(dirnames_annotated)
    process_file(fname_dictionary)
    score(fname_source, fname_corrupted, fname_fixed)


if __name__ == "__main__":
    main()
