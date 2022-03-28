from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from sklearn.metrics import r2_score


def read_plot_data(perplexity_file, data):
    f = open(perplexity_file, "r")
    line = f.readline()
    while line:
        if line.split(",")[0] in data:
            dot_coord = [int(line.split(",")[1]), float(line.split(",")[2][:-1])]
            data[line.split(",")[0]].append(dot_coord)
        else:
            data[line.split(",")[0]] = []
        line = f.readline()
    f.close()


def get_y_on_poly1d(x, y, degree):
    return np.poly1d(np.polyfit(x, y, degree))(x)


def get_degree(x, y):
    degree = 0
    delta = 0.05
    prev_score = -0.1
    score = 0.0
    while score - prev_score > delta:
        degree += 1
        prev_score = score
        score = r2_score(y, get_y_on_poly1d(x, y, degree))
    return degree


def create_plot(perplexity_file):
    data = {}
    read_plot_data(perplexity_file, data)
    plt.style.use('Solarize_Light2')
    colors = list(mcolors.BASE_COLORS)
    for iter_num in data.keys():
        color = next(colors)
        x = list(list(zip(*data[iter_num]))[0])
        y = list(list(zip(*data[iter_num]))[1])
        plt.plot(x, y, '-o', linewidth="0.3", color=color, label="i:" + iter_num)
        plt.plot(x, get_y_on_poly1d(x, y, get_degree(x, y)), linewidth="1", color=color)
    plt.xlabel("Topics num")
    plt.ylabel("Perplexity")
    plt.legend(loc="upper left")
    plt.show()