import nltk
import numpy as np
from text_annotation.text_processor import process_file
from typos_correction.corrupted_text_processor import process_corrupted_file
from text_vectorization.vectorize import build_frequency_dict_and_term_doc_matrix, process_text_for_vectorization, reduce_text_vector
from text_vectorization.w2v import w2v_train, use_model


def main():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')
    # lab 1
    # process_file('../assets/train.csv', 'train')
    # process_file('../assets/test.csv', 'test')
    # lab 2
    # process_corrupted_file("../assets/test-corrupted.csv", "../assets/test")
    # lab 3
    # build_frequency_dict_and_term_doc_matrix("../assets/train", 5, "../assets/train-dict.json", "../assets/train-td-matrix.csv")
    # test = process_text_for_vectorization("crude prices plus worries  about the economy and the outlook for earnings are expected to  hang over the stock market next week during the depth of the  summer doldrums", "../assets/train-dict.json", "../assets/train-td-matrix.csv")
    # print(reduce_text_vector(np.asarray([test], dtype=np.float32).reshape(1, -1), 1))
    # w2v_train("../assets/train.csv")
    use_model()


if __name__ == "__main__":
    main()
