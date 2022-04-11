from text_topic_modelling.model import train_lda_model, use_lda_model
from text_topic_modelling.preprocessing import build_csr_term_doc_matrix
from text_topic_modelling.visualize import create_plot

NUM = 10


def process_topic():
    build_csr_term_doc_matrix(
        "../assets/train",
        5,
        "../assets/train-dict-new.json",
        "../assets/train-td-matrix.npz"
    )
    iterations = [5, 10, 20]
    topics = [10, 20, 30, 40]
    for iteration in iterations:
        print(str(iteration))
        for topic in topics:
            print(str(topic))
            train_lda_model(topic, iteration, "../assets/train-td-matrix.npz")
            use_lda_model(
                "../assets/lda-models/lda_model_" + str(iteration) + "_" + str(topic) + ".jl",
                "../assets/train-dict-new.json",
                "../assets/train-td-matrix.npz",
                "../assets/train.csv",
                "../assets/train-td-matrix.npz",
                NUM
            )
    create_plot("../assets/topic-modelling-results/perplexity")
