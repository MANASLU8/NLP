import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from sklearn.metrics import r2_score


def choose_poly_deg(x, y):
    r2_prev = -0.1
    r2_curr = 0.0
    deg = 0
    while r2_curr > r2_prev:
        deg += 1
        coeffs = np.polyfit(x, y, deg)
        trendpoly = np.poly1d(coeffs)
        y_res = trendpoly(x)
        r2_prev = r2_curr
        r2_curr = r2_score(y, y_res)
    return deg - 1


def create_plot(perplexity_data_path):
    plt.style.use('seaborn-whitegrid')
    f = open(perplexity_data_path, "r")
    plot_data_by_iter_num = {}
    line = f.readline()
    while line:
        if line.split(",")[0] in plot_data_by_iter_num:
            dot_coord = [int(line.split(",")[1]), float(line.split(",")[2][:-1])]
            plot_data_by_iter_num[line.split(",")[0]].append(dot_coord)
        else:
            plot_data_by_iter_num[line.split(",")[0]] = []
        line = f.readline()
    f.close()
    buf = plot_data_by_iter_num.copy()
    for key in sorted(buf.keys()):
        plot_data_by_iter_num[key] = sorted(buf[key])
    color = iter(cm.rainbow(np.linspace(0, 1, len(plot_data_by_iter_num))))
    for iter_num in plot_data_by_iter_num.keys():
        x = list(list(zip(*plot_data_by_iter_num[iter_num]))[0])
        y = list(list(zip(*plot_data_by_iter_num[iter_num]))[1])
        c = next(color)
        plt.plot(x, y, '-o', linewidth="0.5", color=c, label=("iter=" + iter_num))
        deg = choose_poly_deg(x, y)
        p = np.polyfit(x, y, deg)
        trendpoly = np.poly1d(p)
        plt.plot(x, trendpoly(x), linewidth="2", color=c)
    plt.legend(loc="upper left")
    plt.xlabel("Number of topics")
    plt.ylabel("Perplexity")
    plt.show()
