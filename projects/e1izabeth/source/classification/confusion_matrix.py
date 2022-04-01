import math


class ConfusionMatrix:
    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def get_accuracy(self):
        divider = self.TN + self.TP + self.FN + self.FP
        return (self.TN + self.TP) / divider if divider != 0 else math.nan

    def get_precision(self):
        divider = self.TP + self.FP
        return self.TP / divider if divider != 0 else math.nan

    def get_recall(self):
        divider = self.TP + self.FN
        return self.TP / divider if divider != 0 else math.nan

    def get_f1(self):
        precision = self.get_precision()
        recall = self.get_recall()
        return 2 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else math.nan

    def print_metrics(self):
        print("\taccuracy:" + self.get_accuracy())
        print("\tprecision:" + self.get_precision())
        print("\trecall:" + self.get_recall())
        print("\tf1:" + self.get_f1())