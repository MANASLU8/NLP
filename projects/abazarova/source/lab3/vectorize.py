import math
from nltk import collections
from sklearn.decomposition import PCA


def tf_idf(corpus):
    documents_list = []
    for text in corpus:
        tf_idf_vector = []
        computed_tf = compute_tf(text)
        for word in computed_tf:
            tf_idf_vector.append((word, computed_tf[word] * compute_idf(word, corpus)))
        documents_list.append(tf_idf_vector)
    return documents_list


def compute_tf(text):
    # На вход берем текст в виде списка (list) слов
    # Считаем частотность всех терминов во входном массиве с помощью
    # метода Counter библиотеки collections
    tf_text = collections.Counter(text)
    for i in tf_text:
        # для каждого слова в tf_text считаем TF путём деления
        # встречаемости слова на общее количество слов в тексте
        tf_text[i] = tf_text[i]/float(len(text))
    # возвращаем объект типа Counter c TF всех слов текста
    return tf_text


def compute_idf(word, corpus):
    # на вход берется слово, для которого считаем IDF
    # и корпус документов в виде списка списков слов
    return math.log10(len(corpus)/sum([1.0 for i in corpus if word in i]))


def score_tf_idf(tfidf):
    # для предложений, которые есть - считаем среднее по значениям каждой строчки
    # если в строчке какого-то слова нет, то считаем, что оно 0
    # на выходе вектор из всех предложений
    score = {}
    for x in tfidf:
        for word_score in x:
            #print(word_score)
            if word_score[0] not in score:
                score[word_score[0]] = word_score[1]
            else:
                score[word_score[0]] += word_score[1]
    for x in score:
        score[x] /= len(tfidf)
    return score


def resize(vect, size):
    # fit PCA model to the vectors
    pca = PCA(n_components=size)
    result = pca.fit_transform(vect)
    return result
