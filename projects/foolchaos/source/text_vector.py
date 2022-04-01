from text_vectorization.vector_demo import test_w2v_and_tfidf_models, test_different_w2v_models
from text_vectorization.vectorize import build_frequency_dict_and_term_doc_matrix
from text_vectorization.w2v import w2v_train, create_doc_embeddings_file


def process_vectorization():
    build_frequency_dict_and_term_doc_matrix(
        "../assets/train",
        10,
        "../assets/train-dict-improved.json",
        "../assets/train-td-matrix"
    )
    w2v_train("../assets/simple.csv", 5, 100, 5, "../assets/w2v-train-model_100_5.bin")
    w2v_train("../assets/train.csv", 10, 100, 5, "../assets/w2v-train-model_100_10.bin")
    w2v_train("../assets/train.csv", 10, 300, 5, "../assets/w2v-train-model_300_10.bin")
    test_different_w2v_models(
        [
            "../assets/w2v-train-model_100_5.bin",
            "../assets/w2v-train-model_100_10.bin",
            "../assets/w2v-train-model_300_10.bin"
        ]
    )
    test_w2v_and_tfidf_models("../assets/w2v-train-model_100_5.bin", "../assets/train-td-matrix.csv")
    create_doc_embeddings_file(
        "../assets/w2v-train-model_100_5.bin",
        "../assets/test.csv",
        "../assets/train-dict.json",
        "../assets/train-td-matrix.csv",
        "../assets/annotated-corpus/test-embeddings.tsv"
    )
