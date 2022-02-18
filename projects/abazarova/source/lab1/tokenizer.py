import re
from nltk.tokenize import word_tokenize
from nltk import SnowballStemmer
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

node_sep = r'","'

word_regexp = r'^[a-z]+\'?[a-z]*'
big_word_regexp = r'^[A-Za-z]+\'?[a-z]*'
red_word_regexp = r'^[A-Za-z]+(\.[A-Za-z]+)+\.?'
space_regexp = r'[ \\]+'
punct_regexp = r'(\.\.\.|[\"\*=\'_,.!?;:\(\)\-\/\#\$]|\-\-|\"\")+'
site_regexp = r'(https?:\/\/)?([A-Za-z]*\.)+(com|net)(\/[A-Za-z0-9\.]*)*\/?'
money_regexp = r'\$[0-9]*(,?[0-9]{3})*\.?[0-9]*([kbmKBM]|bn| [Bb]illion| [Mm]illion| [Hh]undred| [Tt]housand)?'
numer_regexp = r'(#[0-9]+|\'[0-9]+|[0-9]+(th|nd|rd|st)|No. [0-9]+)'
number_regexp = r'-?[0-9]+.?[0-9]*|([0-9]+ )?[0-9]+(\/?[0-9]+)?'
tag_regexp = r'(\&lt;[^(\&lt;)(\&gt;)]*\&gt;|\&gt;|\&lt;)'
words_w_num_regexp = r'([0-9]+[a-zA-Z]+|[a-zA-Z]+[0-9]+)+'
date_regexp = r'[1-2][0-9]{3}\/[0-1][0-9]\/[0-3][0-9]'
other_regexp = r'.{1}'


def tokenize1(text: str):
    array_of_tokens = word_tokenize(text)
    # print(array_of_tokens)
    return array_of_tokens


def tokenize(file: str):
    regs = []
    regs.append((site_regexp, "SITE"))
    regs.append((tag_regexp, "TAG"))
    regs.append((date_regexp, "DATE"))
    regs.append((money_regexp, "MONEY"))
    regs.append((words_w_num_regexp, "WORDNUM"))
    regs.append((numer_regexp, "№"))
    regs.append((number_regexp, "NUM"))
    regs.append((red_word_regexp, "REDUCT"))
    regs.append((big_word_regexp, "BIG"))
    regs.append((word_regexp, "WORD"))
    regs.append((punct_regexp, "PUNCT"))
    # Шаг 1: делим строку на части - то, что внутри кавычек, получаем текст для анализа
    array_of_nodes = re.compile(node_sep).split(file)
    array_of_nodes[0] = array_of_nodes[0][1:]
    array_of_nodes[-1] = array_of_nodes[-1][:-1]
    # print("***tokenize***NODES")
    # print(*array_of_nodes, sep="\n")
    # return
    file_class = array_of_nodes[0]
    file_name = re.sub(r'[^A-Za-z0-9]+', "_", array_of_nodes[1])
    if file_name[-1] == "_":
        file_name = file_name[:-1]
    text = str(array_of_nodes[1] + ". " + array_of_nodes[2])
    # print(text)
    # Шаг 2: по шаблонам выделяем токены и записываем в список токенов
    array_of_tokens = []
    while len(text) > 0:
        print(text)
        flag = False
        for r in regs:
            if not flag:
                match = re.match(r[0], text)
                if match is not None:
                    print(match.group(0), r[1])
                    text = text[len(match.group(0)):]
                    if match.group(0):
                        array_of_tokens.append(match.group(0))
                    flag = True
        if not flag:
            match = re.match(space_regexp, text)
            if match is not None:
                print(match.group(0), "SPACE")
                text = text[len(match.group(0)):]
                flag = True
        if not flag:
            print(text, "can't read")
            return

    # print(array_of_tokens)
    return array_of_tokens, file_class, file_name


def stemm(word: str):
    stemmer = SnowballStemmer("english")
    stem = stemmer.stem(word)
    print(stem)
    return stem


def lemm(word: str, pos):
    # print(word, pos)
    lemmer = WordNetLemmatizer()
    lem = lemmer.lemmatize(word, pos=get_wordnet_pos(pos))
    print(lem)
    return lem


def pos(words: str):
    pos_tag = nltk.pos_tag(words)
    print(pos_tag)
    return pos_tag


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    # по умолчанию пусть будет существительное
    else:
        return wordnet.NOUN
