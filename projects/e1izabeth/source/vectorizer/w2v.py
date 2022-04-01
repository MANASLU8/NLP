import os
import pandas as pd

from gensim.models import Word2Vec

from tokenizer.tokenizer import tokenize_text, end_of_clause
from utils import eng_stopwords, cosine_distance, pca_reduce


def w2v_train(fname, window, min_freq, vec_size, out_path):
    print('working on ', fname)
    df = pd.read_csv(fname, sep=',', header=None)
    data = df.values
    data_count = len(data)
    n = 0
    training_list = []
    for row in data:
        try:
            for i in range(1, len(row)):
                text = row[i]
                tokens = tokenize_text(text)
                words = []
                prev = [0, '', '']
                for w in tokens:
                    if w[1] == 'word' and w[1] not in eng_stopwords:
                        words.append(w[2].strip().lower())
                    elif prev[2] in end_of_clause:
                        training_list.append(words)
                        words = []
                    prev = w
                if len(words) > 0:
                    training_list.append(words)
                    words = []  # cleanup list of words
        except Exception as e:
            print(e)
            pass
        n = n + 1
        if n % 1000 == 0:
            print(int(n * 100 / data_count), '%')
    print("Training")
    model = Word2Vec(sentences=training_list, window=window, min_count=min_freq, workers=4, vector_size=vec_size)
    model.save(out_path)


def apply_w2v_model(model_path, initial_token, similar_tokens, same_tokens, other_tokens, vec_dimension):
    model = Word2Vec.load(model_path)
    print(model)
    tokens = []
    print("Token: " + initial_token)
    tokens.append(initial_token)
    print("Similar:")
    print_cosine_distance(model, tokens, initial_token, similar_tokens)
    print("Same:")
    print_cosine_distance(model, tokens, initial_token, same_tokens)
    print("Other:")
    print_cosine_distance(model, tokens, initial_token, other_tokens)

    x = [model.wv.get_vector(token) for token in tokens]
    result = pca_reduce(x, vec_dimension)
    return result, tokens


def print_cosine_distance(model, tokens, initial_token, curr_tokens):
    for token in curr_tokens:
        tokens.append(token)
        print("\t" + token + ": " + str(
            cosine_distance(model.wv.get_vector(initial_token), model.wv.get_vector(token))))
