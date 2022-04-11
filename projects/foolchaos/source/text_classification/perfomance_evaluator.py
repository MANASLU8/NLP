import logging
import math

from .metrics import metrics

logging.basicConfig(format="%(asctime)s -- %(message)s", level=logging.INFO)


def evaluate_model(svm_model, x_test, y_test):
    true_labels = y_test
    model_output = svm_model.predict(x_test)

    # create confusion matrix for each class
    for label_num, predicted_label in enumerate(model_output):
        true_label = true_labels[label_num]
        if predicted_label == true_label:
            metrics[str(true_label)]["TP"] += 1
            for class_m in metrics:
                if class_m != str(predicted_label):
                    metrics[class_m]["TN"] += 1
        else:
            metrics[str(true_label)]["FN"] += 1
            metrics[str(predicted_label)]["FP"] += 1
            for class_m in metrics:
                if (class_m != str(predicted_label)) and (class_m != str(true_label)):
                    metrics[class_m]["TN"] += 1

    # calculate metrics for each class
    for class_m in metrics:
        metrics_class = metrics[class_m]
        if (metrics_class["TP"] + metrics_class["FP"]) != 0:
            metrics_class["precision"] = (metrics_class["TP"] / (metrics_class["TP"] + metrics_class["FP"]))
        else:
            metrics_class["precision"] = math.nan
        if (metrics_class["TP"] + metrics_class["FN"]) != 0:
            metrics_class["recall"] = (metrics_class["TP"] / (metrics_class["TP"] + metrics_class["FN"]))
        else:
            metrics_class["recall"] = math.nan
        if (metrics_class["TN"] + metrics_class["TP"] + metrics_class["FN"] + metrics_class["FP"]) != 0:
            metrics_class["accuracy"] = (metrics_class["TN"] + metrics_class["TP"]) / \
                                        (metrics_class["TN"] + metrics_class["TP"] + metrics_class["FN"] +
                                         metrics_class["FP"])
            metrics_class["error_rate"] = (metrics_class["FN"] + metrics_class["FP"]) / \
                                          (metrics_class["TN"] + metrics_class["TP"] + metrics_class["FN"] +
                                           metrics_class["FP"])
        else:
            metrics_class["accuracy"] = math.nan
            metrics_class["error_rate"] = math.nan
        if (metrics_class["precision"] + metrics_class["recall"]) != 0:
            metrics_class["f1"] = 2 * ((metrics_class["precision"] * metrics_class["recall"]) /
                                       (metrics_class["precision"] + metrics_class["recall"]))
        else:
            metrics_class["f1"] = math.nan
        logging.info(metrics_class)
    logging.info("\n")
    # calculate final metrics for model
    accuracy = 0.0
    error_rate = 0.0
    precision_macro = 0.0
    recall_macro = 0.0
    precision_micro_numerator = 0
    precision_micro_denominator = 0
    recall_micro_numerator = 0
    recall_micro_denominator = 0
    for class_m in metrics:
        accuracy += metrics[class_m]["accuracy"]
        error_rate += metrics[class_m]["error_rate"]
        precision_macro += metrics[class_m]["precision"]
        recall_macro += metrics[class_m]["recall"]
        precision_micro_numerator += metrics[class_m]["TP"]
        precision_micro_denominator += (metrics[class_m]["TP"] + metrics[class_m]["FP"])
        recall_micro_numerator += metrics[class_m]["TP"]
        recall_micro_denominator += (metrics[class_m]["TP"] + metrics[class_m]["FN"])
    accuracy = accuracy / len(metrics)
    error_rate = error_rate / len(metrics)
    precision_macro = precision_macro / len(metrics)
    recall_macro = recall_macro / len(metrics)
    f1_macro = 2 * ((precision_macro * recall_macro) / (precision_macro + recall_macro))
    precision_micro = precision_micro_numerator / precision_micro_denominator
    recall_micro = recall_micro_numerator / recall_micro_denominator
    f1_micro = 2 * ((precision_micro * recall_micro) / (precision_micro + recall_micro))
    return accuracy, error_rate, recall_micro, recall_macro, precision_micro, precision_macro, f1_micro, f1_macro
