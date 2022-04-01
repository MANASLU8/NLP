from lab2.fixes import *
from lab2.levenst import *
from lab1.reader import *
from lab1.tokenizer import *


def lab2(path_t, path_d, path_c, percent, algo):
    print("Процент: ", percent)
    files = read_from_file(path_t)
    tok_d = read_from_file(path_d)
    cor_files = read_from_file(path_c)
    tokens_correct = []
    num_tok_all = 0
    num_tok_w_typo = 0
    num_tok_cant = 0
    tokens_b4 = []
    for i in range(len(files)):
        file = files[i]
        cor = cor_files[i]
        compare = tokenize(cor[0])
        tokens_correct.append(compare[0])
        given = tokenize(file[0])
        tokens_b4.append(given[0])
        rez = []
        num_tok_all += len(given[0])
        for j in range(len(given[0])):
            if given[0][j] != "\n":
                x = given[0][j]
                y = compare[0][j]
                rez_tok, is_typo = fixtypo(tok_d, x, algo, percent)
                rez.append(rez_tok)
                score = lev_hirsch(rez_tok, y)
                score_b4 = lev_hirsch(x, y)
                if is_typo:
                    if x != ",":
                        print("Опечатка! Было:", x, "Стало:", rez_tok, "Нужно:", y, "Расстояние было:", score_b4,
                              "Расстояние стало:", score, end=" ")
                        if score != 0:
                            print("Не исправить")
                            num_tok_cant += 1
                        else:
                            print("Исправлено")
                        num_tok_w_typo += 1
                elif score_b4 != 0:
                    print("Опечатка не найдена!!! Было:", x, "Нужно:", y, "Расстояние:", score_b4)
                    num_tok_w_typo += 1
                    num_tok_cant += 1

        print("Файл", given[2], "проверен")
        print("Всего токенов:", num_tok_all)
        print("Из них опечаток было:", num_tok_w_typo)
        print("Из них опечаток осталось:", num_tok_cant)
