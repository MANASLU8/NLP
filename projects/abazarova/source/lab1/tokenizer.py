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

comma_regexp = r'","'
sentence_regexp = r'[?!.] '
word_regexp = r'([,:;]? )|(\\+)|-+'


def tokenize1(text: str):
    array_of_tokens=word_tokenize(text)
    # print(array_of_tokens)
    return array_of_tokens


def tokenize(text: str):
    # Шаг 1: делим строку на части - то, что внутри кавычек
    array_of_nodes = re.compile(comma_regexp).split(text)
    # print(*array_of_nodes,sep="\n")
    # Шаг 2: делим каждую ноду на предложения
    array_of_sentences = []
    for txt in array_of_nodes:
        array_of_sentences += re.compile(sentence_regexp).split(txt)
    # print(*array_of_sentences, sep="\n")
    # Шаг 3: делим предложения на слова
    array_of_words = []
    for txt in array_of_sentences:
        array_of_words += re.compile(word_regexp).split(txt)
    # print(*array_of_words, sep="\n")
    # Шаг 4: обрезаем лишние знаки препинания и убираем повторы
    array_of_tokens = []
    for txt in array_of_words:
        # print(txt)
        word = pretty(txt)
        # print(word)
        if word:
            array_of_tokens += [word]
    # print(*array_of_tokens, sep="\n")
    # ? Шаг 5: вернуть знаки препинания и теги??
    # my_tokens = re.findall(_regexp,text)!!!
    return array_of_tokens


def pretty(token: str):
    word = token
    if not word:
        return False

    if len(word) <= 0:
        return False

    # Нужно обработать теги! '&lt;strong&gt;Opinion&lt;/strong&gt' -> 'Opinion'
    # все теги формата: &lt;ТЕГ&gt;
    if "&" in word:
        # print(word)
        tags = []
        tag = ""
        tflag = False
        wflag = False
        for c in range(len(word)):
            if word[c] == "&":
                tags.append(tag)
                tflag = True
                wflag = False
            if tflag and not wflag:
                tag = tag+word[c]
                if tag == "&gt":
                    wflag = True
            if word[c] == ";":
                tflag = False
                wflag = True
                tag = ""
            elif wflag:
                tag = tag+word[c]
        # print(tags)
        if len(tags) == 4:
            word = tags[2]

    while (len(word) > 0) and (not word[-1].isalnum()):
        word = word[:-1]

    while (len(word) > 0) and (not word[0].isalnum()):
        word = word[1:]

    return word


def stemm(word: str):
    stemmer = SnowballStemmer("english")
    stem = stemmer.stem(word)
    # print(stem)
    return stem


def lemm(word: str, pos):
    # print(word, pos)
    lemmer = WordNetLemmatizer()
    lem = lemmer.lemmatize(word, pos=get_wordnet_pos(pos))
    # print(lem)
    return lem


def pos(word: str):
    pos_tag = nltk.pos_tag([word])
    # print(pos_tag)
    pos_tag = pos_tag[0][1]
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
