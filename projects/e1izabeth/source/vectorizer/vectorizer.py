import pandas as pd

import numpy as np
from gensim.models import Word2Vec
from scipy import sparse
from scipy.sparse import csr_matrix

from tokenizer.tokenizer import tokenize_text
from utils import eng_stopwords, calc_tf_idf


class TextVectorizer:
    def __init__(self, token_entries, old_vectorizer=None):
        self.token_entries = token_entries
        self.all_words_total_val = sum(sum(e.values()) for e in token_entries.values())
        if old_vectorizer is None:
            self.all_words = []
            self.index_by_word = dict()
            for word, _ in token_entries.items():
                self.index_by_word[word] = len(self.all_words)
                self.all_words.append(word)
        else:
            self.all_words = old_vectorizer.all_words
            self.index_by_word = old_vectorizer.index_by_word
        self.words_by_file = dict()
        for word, entry in self.token_entries.items():
            for fname, n in entry.items():
                words = self.words_by_file.get(fname)
                if words is None:
                    self.words_by_file[fname] = words = dict()
                words[word] = n
        self.fnames = list(self.words_by_file.keys())
        self.fnames.sort()

    def get_tf_idf_for_text(self, text):
        tokens_list = [token_info[2] for token_info in tokenize_text(text)
                       if (token_info[1] == "word" and token_info[2].lower() not in eng_stopwords)]
        words = dict()
        for word in tokens_list:
            n = words.get(word)
            if n is None:
                n = 0
            words[word] = n + 1

        word_measures = dict()
        for word, n in words.items():
            word_measures[word] = calc_tf_idf(n, float(len(tokens_list)), len(self.token_entries.get(word)))

        text_vector = []
        for i in range(0, len(self.all_words)):
            v = word_measures.get(self.all_words[i])
            text_vector.append(0 if v is None else v)

        return text_vector

    def get_tf_idf_for_word(self, word):
        if word in self.all_words:
            return calc_tf_idf(sum(self.token_entries.get(word).values()), float(self.all_words_total_val), len(self.token_entries.get(word)))
        else:
            return None

    def get_w2v_embedding(self, model, text):
        sentences_vals = []
        sentences_vals_avg = []

        tokens_list = [token_info for token_info in tokenize_text(text)
                       if ((token_info[1] == "word" or token_info[1] == "punct") and token_info[2].lower() not in eng_stopwords)]
        current_sentence_tokens = []
        for token_info in tokens_list:
            token = token_info[2]
            if token_info[1] == "punct":
                if current_sentence_tokens:
                    sentences_vals.append(current_sentence_tokens)
                current_sentence_tokens = []
            else:
                if token in list(model.wv.index_to_key):
                    tf_idf = self.get_tf_idf_for_word(token)
                    if tf_idf is not None:
                        weighted_vector = model.wv.get_vector(token) * tf_idf
                        current_sentence_tokens.append(weighted_vector)
        if not sentences_vals:
            sentences_vals.append(current_sentence_tokens)
        for sentence in sentences_vals:
            sentences_vals_avg.append(np.average(sentence, axis=0))
        document_vector = np.average(sentences_vals_avg, axis=0)
        return document_vector

    def get_w2v_embeddings(self, model_path, file_to_process, output_file):
        model = Word2Vec.load(model_path)
        f_out = open(output_file, "w+")

        df = pd.read_csv(file_to_process, sep=',', header=None)
        data = df.values
        data_count = len(data)
        n = 0
        for row in data:
            n = n + 1
            try:
                if len(row) > 1:
                    text = row[1]
                    for i in range(2, len(row)):
                        text = text + '. ' + row[i]

                    document_vector = self.get_w2v_embedding(model, text)
                    f_out.write(str(n) + "\t" + "\t".join(str(component) for component in document_vector))
                    f_out.write("\n")
            except Exception as e:
                print(e)
                print([n, text, document_vector])
                pass
            if n % 50 == 0:
                print(n * 100 / data_count, '%', ' (', n , '/', data_count , ')')
                f_out.flush()
        f_out.close()

    def get_csr_matrix(self, min_fr):
        indptr = [0]
        indices = []
        data = []
        for file in range(0, len(self.fnames)):
            for word, n in self.words_by_file.get(self.fnames[file]).items():
                if n >= min_fr:
                    wi = self.index_by_word.get(word)
                    indices.append(wi)
                    data.append(1)
            indptr.append(len(indices))
        return csr_matrix((data, indices, indptr), dtype=int, shape=(len(self.fnames), len(self.all_words)))

    def save_matrix(self, min_frequency, matrix_path):
        sparse.save_npz(matrix_path, self.get_csr_matrix(min_frequency))
