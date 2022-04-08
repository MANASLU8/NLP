import csv
import os

forbidden_words = set()


def read_forbidden_words(path: str):
    with open(path) as file:
        reader = csv.reader(file)
        for record in reader:
            word = record[0]
            if word not in forbidden_words:
                forbidden_words.add(word)

    return forbidden_words


def read_directory(path: str):
    result = []
    for file in os.listdir(path):
        abs_path = os.path.join(path, file)
        if not os.path.isdir(abs_path):
            result.append(abs_path)
        else:
            file_paths = read_directory(abs_path)
            result.extend(file_paths)

    return result


def read_corpus(dir_path: str):
    paths = read_directory(dir_path)

    corpus = []
    for path in paths:
        with open(path) as file:
            words = []
            reader = csv.reader(file, delimiter='\t')
            for record in reader:
                if not record:
                    continue

                _, lemma, = record
                word = lemma.lower()
                words.append(word)
            corpus.append(words)

    return corpus
