import click

from lab1.tokenizer import *
from lab1.reader import *
from lab2.fixes import *

@click.group()
def main():
    pass


@main.command()
@click.argument("paths", type=str)
@click.argument("pos_tags", type=bool, required=False)
def token(paths, pos_tags=False):
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


@main.command()
@click.argument("path_t", type=str)
@click.argument("path_d", type=str)
@click.argument("path_c", type=str)
@click.argument("algo", type=str, required=False)
def typo(path_t, path_d, path_c, algo):
    files = read_from_file(path_t)
    tok_d = read_from_file(path_d)
    cor_files = read_from_file(path_c)
    tokens_correct = []
    for file in cor_files:
        tokens_correct.append(tokenize(file[0])[0])
    num_tok_all = 0
    num_tok_w_typo = 0
    num_tok_cant = 0
    tokens_b4 = []
    tokens_after = []
    for file in files:
        given = tokenize(file[0])
        tokens_b4.append(given[0])
        rez = []
        num_tok_all += len(given[0])
        for x in given[0]:
            rez_tok, is_typo, cant_fix = fixtypo(tok_d, x, algo)
            rez.append(rez_tok)
            if is_typo:
                num_tok_w_typo += 1
            elif cant_fix:
                num_tok_cant += 1

        print("Всего токенов: ", num_tok_all)
        print("Из них опечаток было: ", num_tok_w_typo)
        print("Из них опечаток осталось: ", num_tok_cant)
        write_typos(path_c, given[1], given[2], rez)
        tokens_after.append(rez)
    for i in range(len(tokens_correct)):
        for j in range(len(tokens_correct[i])):
            if tokens_correct[i][j] != "\n":
                print(tokens_correct[i][j], end="|")
                if j < len(tokens_b4[i]):
                    print(tokens_b4[i][j], end="|")
                else:
                    print("   |", end="")
                if j < len(tokens_after[i]):
                    print(tokens_after[i][j])
                else:
                    print()


if __name__ == "__main__":
    main()
