from text_classification.svm_models import *

POLY = 3


def process_classification():
    iterations = [200, 400, 800, 1000, 8000, 10000, 20000, 30000, 40000, 50000, 60000, 80000]
    for iteration in iterations:
        # rbf
        train_svm_rbf_kernel(
            "../assets/train.csv",
            "../assets/annotated-corpus/test-embeddings.tsv",
            iteration,
            "svm_rbf"
        )
        use_svm_model(
            "../assets/svm-models/rbf_" + str(iteration) + ".jl",
            "../assets/test.csv",
            "../assets/annotated-corpus/test-embeddings.tsv",
            "../assets/svm-models-evaluation/svm_rbf",
            iteration
        )
        # linear
        train_svm_linear_kernel(
            "../assets/train.csv",
            "../assets/annotated-corpus/test-embeddings.tsv",
            iteration,
            "svm_linear"
        )
        use_svm_model(
            "../assets/svm-models/linear_" + str(iteration) + ".jl",
            "../assets/test.csv",
            "../assets/annotated-corpus/test-embeddings.tsv",
            "../assets/svm-models-evaluation/svm_linear",
            iteration
        )
        # sigmoid
        train_svm_sigmoid_kernel(
            "../assets/train.csv",
            "../assets/annotated-corpus/test-embeddings.tsv",
            iteration,
            "svm_sigmoid"
        )
        use_svm_model(
            "../assets/svm-models/sigmoid_" + str(iteration) + ".jl",
            "../assets/test.csv",
            "../assets/annotated-corpus/test-embeddings.tsv",
            "../assets/svm-models-evaluation/svm_sigmoid",
            iteration
        )
        # poly
        train_svm_poly_kernel(
            "../assets/train.csv",
            "../assets/annotated-corpus/test-embeddings.tsv",
            iteration,
            "svm_poly3",
            POLY
        )
        use_svm_model(
            "../assets/svm-models/poly3_" + str(iteration) + ".jl",
            "../assets/test.csv",
            "../assets/annotated-corpus/test-embeddings.tsv",
            "../assets/svm-models-evaluation/svm_poly3",
            iteration
        )

    # experiments
    vector_dims = [100, 85, 70, 60, 55, 30, 20, 15, 5]
    for dim in vector_dims:
        use_model_with_dim_reduction_scale_after(
            "../assets/svm-models/rbf50000.jl",
            dim,
            "svm_rbf50000_reduced",
            "../assets/test.csv",
            "../assets/annotated-corpus/test-embeddings.tsv",
            "../assets/train.csv",
            "../assets/annotated-corpus/test-embeddings.tsv"
        )
