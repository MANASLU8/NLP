from lab3.corpus import corpus
from lab3.neur import neur_train, compare_tokens
from lab3.algo_vec import algo_vec, make_vector_file
from lab3.tdm import tdm
from lab1.reader import *
import json


def lab3(train_path, test_path, vec_path, count_words, fdict_file, tdm_file):

    train = read_from_file(train_path)

    dictionary = {}
    all_sum = 0
    doc_count = 0
    for doc in train:
        tmp = corpus(doc)
        doc_count += 1
        for sent in tmp:
            for token in sent:
                # print(token)
                all_sum += 1
                if token not in dictionary:
                    dictionary[token] = {}
                if doc_count not in dictionary[token]:
                    dictionary[token][doc_count] = 1
                else:
                    dictionary[token][doc_count] += 1
    print("fdict finished")
    # обрезаем самые редкие слова, например, если из 100 слов это встречается
    #  только percentage_words и меньше раз, то удаляем. percentage_words вводится как арг кс.
    print("всего слов:", all_sum)
    print("в результате слово должно встречаться минимум", count_words, "раз")
    fdict = {}
    for word in dictionary:
        summa = 0
        for doc in dictionary[word].keys():
            summa += dictionary[word][doc]
        if summa >= count_words:
            fdict[word] = summa

    with open(fdict_file, 'w') as file_dict:
        json.dump(fdict, file_dict)

    tdm(tdm_file, dictionary, doc_count)

    print("tdm finished")
    # PART 2 - vectorize.py
    # PART 3 - neur.py
    # PART 4 - cos_dist.py
    # Part 5 - vectorize.py
    # PART 6
    # PART 7 - algo_vec.py

    # Part 8:
    mod_path = str(Path(str(Path(Path.cwd()))[:-len("source")], "assets", "model.bin"))
    neur_train(train_path, mod_path, count_words)
    print("model trained")
    compare_tokens(mod_path)
    print("tokens compared")
    make_vector_file(mod_path, test_path, train_path, vec_path)
    print("lab finished")
