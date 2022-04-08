import json

from scipy import sparse
from scipy.sparse import csr_matrix

from lab1.reader import read_from_file
from lab3.corpus import corpus


def get_csr_matrix(corp, dictionary):
    # print(corp)
    vocabulary = {}
    ptr = [0]
    indexes = []
    data = []
    for d in corp:
        for token in d:
            if token in dictionary:
                index = vocabulary.setdefault(token, len(vocabulary))
                indexes.append(index)
                data.append(1)
        ptr.append(len(indexes))
    return csr_matrix((data, indexes, ptr), dtype=int)


def make_csr(train_path, csr_path, count_words):
    train = read_from_file(train_path)
    dictionary = {}
    doc_count = 0
    for doc in train:
        tmp = corpus(doc)
        doc_count += 1
        for sent in tmp:
            for token in sent:
                # print(token)
                if token not in dictionary:
                    dictionary[token] = {}
                if doc_count not in dictionary[token]:
                    dictionary[token][doc_count] = 1
                else:
                    dictionary[token][doc_count] += 1
    fdict = {}
    for word in dictionary:
        summa = 0
        for doc in dictionary[word].keys():
            summa += dictionary[word][doc]
        if summa >= count_words:
            fdict[word] = summa

    corp = []
    for line in train:
        corp += corpus(line)
    # print(corp)
    td_matrix = get_csr_matrix(corp, fdict)
    sparse.save_npz(csr_path, td_matrix)
