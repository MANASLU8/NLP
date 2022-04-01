import joblib
from datetime import datetime as dt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from classification.confusion_matrix import ConfusionMatrix


class SVM:
    def __init__(self, svm_model, dim=None):
        self.svm_model = svm_model
        self.training_time = 0.0
        self.pca = None
        self.scaler = None
        self.dim = 100
        if dim is not None:
            self.pca = PCA(n_components=dim)
            self.scaler = StandardScaler()
            self.dim = dim

    def train(self, x_train, y_train):
        if self.dim is not None:
            x_train = self.pca.fit_transform(x_train, y_train)
            x_train = self.scaler.fit_transform(x_train, y_train)
        start_time = dt.now()
        self.svm_model.fit(x_train, y_train)
        self.training_time = (dt.now() - start_time).seconds

    def save_model(self, path):
        joblib.dump(self.svm_model, path)

    def apply(self, x_test, y_test, classes_num, evaluation_results_file):
        if self.dim is not None:
            x_test = self.pca.fit_transform(x_test, y_test)
            x_test = self.scaler.fit_transform(x_test, y_test)
        accuracy, recall, precision, f1 = self.evaluate(x_test, y_test, classes_num)

        self.append_result(evaluation_results_file, accuracy, recall, precision, f1)
        print("\t" + str(accuracy) + "\t" + str(recall) + "\t" + str(precision) + "\t" + str(f1) + "\n")

    def evaluate(self, x_test, y_test, classes_num):
        prediction = self.svm_model.predict(x_test)
        cm = [ConfusionMatrix()] * classes_num

        for label_num, predicted_label in enumerate(prediction):
            label = y_test[label_num]
            if predicted_label == label:
                cm[label - 1].TP += 1
                for class_num in range(classes_num):
                    if class_num != str(predicted_label):
                        cm[class_num - 1].TN += 1
            else:
                cm[label - 1].FP += 1
                cm[predicted_label - 1].FN += 1
                for class_num in range(classes_num):
                    if (class_num != str(predicted_label)) and (class_num != str(label)):
                        cm[class_num - 1].TN += 1

        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        for class_num in range(classes_num):
            accuracy += cm[class_num - 1].get_accuracy()
            precision += cm[class_num - 1].get_precision()
            recall += cm[class_num - 1].get_recall()
        accuracy = accuracy / classes_num
        precision = precision / classes_num
        recall = recall / classes_num
        f1 = 2 * ((precision * recall) / (precision + recall))
        return accuracy, recall, precision, f1

    def save_result(self, evaluation_results_file, max_iter_num):
        eval_file = open(evaluation_results_file, "a+")
        eval_file.write("iter\ttrain time\taccuracy\trecall (micro)\t"
                        "recall (macro)\tprecision (micro)\tprecision (macro)\tf1 (micro)\tf1 (macro)\n")
        eval_file.write(str(max_iter_num) + "\t" + str(self.training_time) + "\n")
        eval_file.close()

    def append_result(self, evaluation_results_file, iter_num, accuracy, recall_micro, recall_macro, precision_micro,
                      precision_macro, f1_micro, f1_macro):
        with open(evaluation_results_file, 'r') as file:
            data = file.readlines()
        for i in range(0, len(data)):
            if (data[i].startswith((str(iter_num) + "\t"))) or (data[i].startswith((str(iter_num) + " "))):
                data[i] = data[i][:-1] + "\t" + str(accuracy) + "\t" + str(recall_micro) + "\t" + \
                          str(recall_macro) + "\t" + str(precision_micro) + "\t" + str(precision_macro) + "\t" + \
                          str(f1_micro) + "\t" + str(f1_macro) + "\n"
        with open(evaluation_results_file, 'w') as file:
            file.writelines(data)
