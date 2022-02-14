import click

from projects.abazarova.source.lab1.tokenizer import *
from projects.abazarova.source.lab1.reader import *


@click.group()
def main():
    pass


@main.command()
@click.argument("paths", type=str)
@click.argument("pos_tags", type=bool)
def token(paths, pos_tags):
    lines = read_from_file(paths)
    # print(lines)
    tokens = set()
    for line in lines:
        tokens |= set(tokenize(line[0]))
    # print(tokens)
    dictionary=dict()
    # Для корректной работы лемматайзера, нужно получить POS-теги
    for t in tokens:
        dictionary[t] = [pos(t)]
    # Получаем стеммы - используем токены, которые получили перед этим
    for t in dictionary:
        dictionary[t].append(stemm(t))
    # Получаем леммы - тут надо использовать теги, которые мы до этого получили
    for t in tokens:
        dictionary[t].append(lemm(t, dictionary[t][0]))

    # for t in dictionary:
        # print(t, dictionary[t])
    write_to_file(paths, dictionary, pos_tags)


if __name__ == "__main__":
    main()
