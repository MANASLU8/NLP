import os
from os import listdir

import pandas as pd
from utils import pca_reduce, cosine_distance, create_plot, eng_stopwords
from vectorizer import TextVectorizer
from w2v import apply_w2v_model, w2v_train

dirnames_annotated = ['../assets/train/Business', '../assets/train/World', '../assets/train/Sports',
                      '../assets/train/Sci-Tech']


def construct_entry_info(tokens_entry):
    for dir in dirnames_annotated:
        print('processing ', dir)
        ff = listdir(dir)
        n = 0
        for file in ff:
            n = n + 1
            if int(n % 100) == 0:
                print('\t', n, '/', len(ff))

            filename = dir + '/' + file
            try:
                df = pd.read_csv(filename, delimiter="\t", header=None, usecols=[0, 1], engine='python', error_bad_lines=False)
                for index, item in df.iterrows():
                    word = str(item[1].lower()).strip()
                    if (word not in eng_stopwords) and (item[0] == 'word'):
                        token = word
                        tentry = tokens_entry.get(token)
                        if tentry is None:
                            tentry = dict()
                            tokens_entry[token] = tentry
                        tfentry = tentry.get(filename)
                        if tfentry is None:
                            tfentry = 0
                        tentry[filename] = tfentry + 1
            except Exception as e:
                print('error while reading  ', filename)
                print(str(e))
                pass


def write_tokens_freq_dictionary(tokens_entry, vec_dict_file_path):
    with open(vec_dict_file_path, "w") as vec_dict_file:
        for word, tentry in tokens_entry.items():
            vec_dict_file.write(word)
            vec_dict_file.write('\n')
            for fname, count in tentry.items():
                vec_dict_file.write(':')
                vec_dict_file.write(fname)
                vec_dict_file.write('\n')
                vec_dict_file.write('-')
                vec_dict_file.write(str(count))
                vec_dict_file.write('\n')
            vec_dict_file.flush()


def read_freq_dict(fname):
    sz = os.path.getsize(fname)
    tokens_entry = dict()
    tentry = None
    filename = None
    with open(fname, "r") as f:
        line = f.readline()
        n = 0
        while line:
            n = n + 1
            if n % 10000 == 0:
                print('\t', f.tell() / sz * 100, '%')
            token = line.strip()

            if token.startswith(":"):
                filename = token[1:]
            elif token.startswith("-"):
                tentry[filename] = int(token[1:])
            else:
                tentry = tokens_entry.get(token)
                if tentry is None:
                    tentry = dict()
                    tokens_entry[token] = tentry

            line = f.readline()

    return tokens_entry


def apply_tfidf_model(tokens_entry, token_to_test, similar_tokens, same_field_tokens, other_tokens, vec_dimension):
    print("Tf-idf")
    tokens = [token_to_test]
    tokens = tokens + similar_tokens + same_field_tokens + other_tokens

    vectorizer = TextVectorizer(tokens_entry)
    x = [vectorizer.get_tf_idf_for_text(token) for token in tokens]
    reduced_embeddings = pca_reduce(x, vec_dimension)
    print("Cosine distance " + tokens[0])
    for i in range(1, len(tokens)):
        print("\t " + tokens[i] + ": " + str(cosine_distance(reduced_embeddings[0], reduced_embeddings[i])))
    return reduced_embeddings, tokens


def demonstrate_w2v_and_tfidf_models(model_path, tokens_entry, vec_dimension):
    initial_tokens = ["basketball", "company", "sudan",]
    similar_tokens = [["football", "soccer"], ["firm", "business", "agency"], ["california", "boston", "chicago"]]
    same_field_tokens = [["sport", "players"], ["bank", "economy"], ["north", "congress"]]
    other_tokens = [["yesterday", "fresh", "rules"], ["re-election", "press", "victory"], ["jail", "sentence", "conviction"]]
    for token in initial_tokens:
        reduced_embeddings, all_tokens = apply_w2v_model(model_path, token, similar_tokens[initial_tokens.index(token)],
            same_field_tokens[initial_tokens.index(token)], other_tokens[initial_tokens.index(token)], vec_dimension)
        create_plot("w2v - " + token, reduced_embeddings, all_tokens)

        reduced_embeddings, all_tokens = apply_tfidf_model(tokens_entry, token, similar_tokens[initial_tokens.index(token)],
            same_field_tokens[initial_tokens.index(token)], other_tokens[initial_tokens.index(token)], vec_dimension)
        create_plot("tf-idf - " + token, reduced_embeddings, all_tokens)


def main():
    vec_dict_file_path = "../assets/vec_dict"
    tokens_entry = []
    construct_entry_info(tokens_entry)
    write_tokens_freq_dictionary(tokens_entry, vec_dict_file_path, 2)
    tokens_entries = read_freq_dict(vec_dict_file_path)
    model_path = '../assets/w2v-train-model_5_100_5.bin'
    w2v_train('../assets/raw-dataset/train.csv', 10, 100, 5, model_path)
    demonstrate_w2v_and_tfidf_models(model_path, tokens_entries, 2)
    vectorizer = TextVectorizer(tokens_entries)
    vectorizer.get_w2v_embeddings(model_path, "../assets/raw-dataset/test.csv", "../assets/annotated-corpus/test-embeddings.tsv")
    #print(tokens_entries)


if __name__ == "__main__":
    main()
