import nltk
from text_annotation.text_processor import process_file
from typos_correction.corrupted_text_processor import process_corrupted_file
from text_vectorization.vectorize import build_frequency_dict_and_term_doc_matrix, vectorize_custom_text
from text_vectorization.w2v import w2v_train, use_w2v_model, create_doc_embeddings_file
from text_vectorization.vector_demo import test_w2v_and_tfidf_models, test_different_w2v_models


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
    build_frequency_dict_and_term_doc_matrix("../assets/train", 5, "../assets/train-dict.json", "../assets/train-td-matrix.csv")
    w2v_train("../assets/train.csv", 5, 100, 5, "../assets/w2v-train-model_100_5.bin")
    w2v_train("../assets/train.csv", 10, 100, 5, "../assets/w2v-train-model_100_10.bin")
    w2v_train("../assets/train.csv", 10, 300, 5, "../assets/w2v-train-model_300_10.bin")
    test_different_w2v_models(["../assets/w2v-train-model_100_5.bin", "../assets/w2v-train-model_100_10.bin", "../assets/w2v-train-model_300_10.bin"])
    test_w2v_and_tfidf_models("../assets/w2v-train-model_100_5.bin", "../assets/train-td-matrix.csv")
    create_doc_embeddings_file("../assets/w2v-train-model_100_5.bin",
                               "../assets/test.csv",
                               "../assets/train-dict.json",
                               "../assets/train-td-matrix.csv",
                               "../assets/annotated-corpus/test-embeddings.tsv")


if __name__ == "__main__":
    main()
