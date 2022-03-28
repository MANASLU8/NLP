import os

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classification.svm import SVM


def get_svm_model_path(kernel_type, iterations):
    return "../../assets/svm_model_" + str(kernel_type) + "_" + str(iterations) + ".jl"


def get_svm_model_path_dim(kernel_type, iterations, dim):
    return "../../assets/svm_model_" + str(kernel_type) + "_" + str(iterations) + "_" + str(dim) + ".jl"


def get_svm_results_path(kernel_type, iterations):
    return "../../assets/svm-models-evaluation/" + str(kernel_type) + "_" + str(iterations)


def get_svm_results_path_dim(kernel_type, iterations, dim):
    return "../../assets/svm-models-evaluation/" + str(kernel_type) + "_" + str(iterations) + "_" + str(dim)


def get_features_and_labels_data(data_file, data_doc_embeddings_file):
    x = []
    y = []
    data = open(data_file, "r")
    embeddings = open(data_doc_embeddings_file, "r")

    for data_line, embeddings_line in zip(data, embeddings):
        label = int(data_line.split('","')[0][1:])
        y.append(label)
        doc_embedding = np.array(embeddings_line[:-1].split("\t"))
        x.append(doc_embedding.astype(float))
    data.close()
    embeddings.close()
    return x, y


def main():
    data_train_path = "../../assets/raw-dataset/train.csv"
    data_test_path = "../../assets/raw-dataset/test.csv"

    emb_train_path = "../../assets/annotated-corpus/train-embeddings.tsv"
    emb_test_path = "../../assets/annotated-corpus/test-embeddings.tsv"

    if not os.path.exists("../../assets/svm-models-evaluation/"):
        os.makedirs("../../assets/svm-models-evaluation/")

    iterations_num_list = [1000, 5000, 10000, 25000, 50000, 75000]
    kernel_types = ["linear", "rbf", "sigmoid"]
    classes_num = 4

    for i in iterations_num_list:
        for kernel_type in kernel_types:
            svm_model = SVC(decision_function_shape='ovo', verbose=True, random_state=0, kernel=kernel_type, max_iter=i)
            svm = SVM(svm_model)
            x_train, y_train = get_features_and_labels_data(data_train_path, emb_train_path)
            svm.train(x_train, y_train)
            svm.save_result(get_svm_results_path(kernel_type, i), i)
            svm.save_model(get_svm_model_path(kernel_type, i))
            x_test, y_test = get_features_and_labels_data(data_test_path, emb_test_path)
            svm.apply(x_test, y_test, classes_num, get_svm_results_path(kernel_type, i))

    dimensions = [25, 4]
    kernel_type = "sigmoid"
    iter_num = 100
    for dim in dimensions:
        svm_model = SVC(decision_function_shape='ovo', verbose=True, random_state=0, kernel=kernel_type, max_iter=iter_num)
        svm = SVM(svm_model, dim)
        x_train, y_train = get_features_and_labels_data(data_train_path, emb_train_path)
        svm.train(x_train, y_train)
        svm.save_result(get_svm_results_path_dim(kernel_type, iter_num, dim), iter_num)
        svm.save_model(get_svm_model_path_dim(kernel_type, iter_num, dim))
        x_test, y_test = get_features_and_labels_data(data_test_path, emb_test_path)
        svm.apply(x_test, y_test, classes_num, get_svm_results_path_dim(kernel_type, iter_num, dim))

    print("main")


if __name__ == "__main__":
    main()
