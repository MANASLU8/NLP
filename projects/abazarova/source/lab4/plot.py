from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from sklearn.metrics import r2_score
# После выполнения экспериментов построить график изменения perplexity в зависимости от количества тем,
# построить полиномиальную аппроксимацию полученного графика, количество элементов полинома выбрать с
# использованием метрики r-squared


def polynom(x, y):
    r2_b4 = -0.1
    r2_now = 0.0
    deg = 0
    while r2_now > r2_b4:
        deg += 1
        coefs = np.polyfit(x, y, deg)
        trendpoly = np.poly1d(coefs)
        y_rez = trendpoly(x)
        r2_b4 = r2_now
        r2_now = r2_score(y, y_rez)
    return deg-1


def build_plot(perp_path):
    plt.style.use('seaborn-whitegrid')
    with open(perp_path, "r") as fin:
        plot_data = {}
        line = fin.readline()
        while line:
            if line.split(",")[0] in plot_data:
                dot_coord = [int(line.split(",")[1]), float(line.split(",")[2][:-1])]
                plot_data[line.split(",")[0]].append(dot_coord)
            else:
                plot_data[line.split(",")[0]] = []
            line = fin.readline()

    color = iter(cm.rainbow(np.linspace(0, 1, len(plot_data))))
    for key in plot_data.keys():
        x = list(list(zip(*plot_data[key]))[0])
        y = list(list(zip(*plot_data[key]))[1])
        c = next(color)
        plt.plot(x, y, '-o', linewidth="0.5", color=c, label=("iter=" + key))
        deg = polynom(x, y)
        p = np.polyfit(x, y, deg)
        trendpoly = np.poly1d(p)
        plt.plot(x, trendpoly(x), linewidth="2", color=c)
    plt.legend(loc="upper left")
    plt.xlabel("Number of topics")
    plt.ylabel("Perplexity")
    plt.show()
