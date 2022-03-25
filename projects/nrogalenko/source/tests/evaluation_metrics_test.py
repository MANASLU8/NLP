import unittest
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from source.text_classification.perfomance_evaluator import evaluate_model
from source.text_classification.svm_models import get_features_and_labels_data
from sklearn.preprocessing import StandardScaler


class SVMModelEvaluationTest(unittest.TestCase):

    def test_model_evaluation(self):
        svm_model = joblib.load("../../assets/svm-models/svm_model_rbf1000.jl")

        x_test, y_test = get_features_and_labels_data("../../assets/test-half.csv",
                                                      "../../assets/annotated-corpus/test-embeddings-half.tsv")
        trans = StandardScaler()
        x_test = trans.fit_transform(x_test)

        accuracy, error_rate, recall_micro, recall_macro,\
        precision_micro, precision_macro, f1_micro, f1_macro = evaluate_model(svm_model, x_test, y_test)

        y_true = y_test
        y_pred = svm_model.predict(x_test)
        self.assertEqual(round(precision_score(y_true, y_pred, average="micro"), 6),
                         round(precision_micro, 6))
        self.assertEqual(round(precision_score(y_true, y_pred, average="macro"), 6),
                         round(precision_macro, 6))
        self.assertEqual(round(recall_score(y_true, y_pred, average="micro"), 6),
                         round(recall_micro, 6))
        self.assertEqual(round(recall_score(y_true, y_pred, average="macro"), 6),
                         round(recall_macro, 6))
        self.assertEqual(round(f1_score(y_true, y_pred, average="micro"), 6),
                         round(f1_micro, 6))


if __name__ == '__main__':
    unittest.main()
