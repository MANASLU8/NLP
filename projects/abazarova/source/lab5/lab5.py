from pathlib import Path

from lab5.svm import *


def lab5(train_path, train_vector_path, test_path, test_vector_path):
    path_to_svm = Path(str(Path(Path.cwd()))[:-len("source")], "assets",
                       "svm")
    counts = [100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 80000]
    for count in counts:
        svm_train("rbf", train_path, train_vector_path, count, "svm_rbf")
        svm_fit(str(Path(path_to_svm, "svm_rbf")) + str(count) + ".jl",
                test_path, test_vector_path, str(Path(path_to_svm, "svm_rbf_rez")), count)

        svm_train("linear", train_path, train_vector_path, count, "svm_linear")
        svm_fit(str(Path(path_to_svm, "svm_linear")) + str(count) + ".jl",
                test_path, test_vector_path, str(Path(path_to_svm, "svm_linear_rez")), count)

        svm_train("sigmoid", train_path, train_vector_path, count, "svm_sigmoid")
        svm_fit(str(Path(path_to_svm, "svm_sigmoid")) + str(count) + ".jl",
                test_path, test_vector_path, str(Path(path_to_svm, "svm_sigmoid_rez")), count)

    # эксперимертируем с размерностью векторов
    vector_dims = [30, 15, 5]
    for dim in vector_dims:
        svm_fir_dimension_redused_after("../assets/svm/svm_rbf50000.jl", dim, "svm_rbf50000_red",
                                        test_path,
                                        test_vector_path,
                                        train_path,
                                        train_vector_path)
        svm_fit_dimensions_redusedb4("../assets/svm/svm_rbf50000.jl", dim, "svm_rbf50000_red_before",
                                     test_path,
                                     test_vector_path,
                                     train_path,
                                     train_vector_path)
