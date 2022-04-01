from lab1.tokenizer import *
from lab1.reader import *


def lab1(paths, pos_tags):
    files = read_from_file(paths)
    # print(files)
    tokens = []
    tokD = set()
    for file in files:
        rez = tokenize(file[0])
        # file_class, file_name, array_of_tokens
        tokens.append([rez[1], rez[2], rez[0]])
        for x in rez[0]:
            if x != '\n':
                tokD |= {x}
    # print(tokens)
    print("TOKENS FINISHED")
    # Для корректной работы лемматайзера, нужно получить POS-теги
    for tok in tokens:
        tok[2] = pos(tok[2])
    print("POS FINISHED")
    # Получаем стеммы - используем токены, которые получили перед этим
    for tok in tokens:
        stem = []
        for i in range(len(tok[2])):
            stem.append(stemm(tok[2][i][0]))
        tok.append(stem)
    print("STEMS FINISHED")
    # Получаем леммы - тут надо использовать теги, которые мы до этого получили
    for tok in tokens:
        lem = []
        for i in range(len(tok[2])):
            lem.append(lemm(tok[2][i][0], tok[2][i][1]))
        tok.append(lem)
    print("LEMS FINISHED")
    # for tok in tokens:
    # print("**********")
    #   for i in range(len(tok[2])):
    #       print(tok[2][i][0], tok[3][i], tok[4][i])

    # print(*tokens, sep="\n")

    write_tokens(paths, tokens, pos_tags)

    create_dict("dict.csv", tokD)
