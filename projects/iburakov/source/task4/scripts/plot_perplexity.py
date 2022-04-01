import json

import pandas as pd
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial
from pandas import DataFrame
from sklearn.metrics import r2_score

from dirs import lda_experiments_dir


def _plot_perplexity_1d(meta_df: DataFrame):
    iterations_space = sorted(meta_df.n_iterations.unique())
    plt.Figure()
    fig, axs = plt.subplots(len(iterations_space), 1, sharex=True, sharey=True, figsize=(5, 3 * len(iterations_space)))
    for ax, n_iterations in zip(axs, iterations_space):
        ax: plt.Axes

        df10 = meta_df[meta_df.n_iterations == n_iterations] \
            .set_index("n_topics") \
            .drop("n_iterations", axis=1) \
            .sort_index()

        x = df10.index.values
        y = df10.test_perplexity.values

        polys = pd.DataFrame(columns=["poly", "r2"])
        polys.r2 = polys.r2.astype(float)
        for poly_deg in range(1, 6):
            # noinspection PyTypeChecker
            poly: Polynomial = Polynomial.fit(x, y, poly_deg)
            # noinspection PyCallingNonCallable
            polys.loc[poly_deg, :] = [
                poly,
                float(r2_score(y, poly(x)))
            ]
        polys = polys.reset_index().rename({"index": "deg"}, axis=1)

        ax.plot(df10.index, df10.values, "--", alpha=0.6, label="perplexity")
        best_poly = polys.iloc[polys.r2.argmax()]
        ax.plot(
            *best_poly.poly.linspace(),
            label=f"poly{best_poly.deg}, r2={best_poly.r2:.2f}"
        )

        ax.set_title(f"n_iterations={n_iterations}")
        ax.set_xlabel("n_topics")
        ax.legend()

    plt.subplots_adjust(hspace=0.3)
    plt.show()


def _plot_perplexity_2d(meta_df: DataFrame):
    df = meta_df.pivot(index="n_iterations", columns="n_topics", values="test_perplexity")
    plt.imshow(-df.values, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(df.columns)), df.columns)
    plt.xlabel("n_topics")
    plt.yticks(range(len(df.index)), df.index)
    plt.ylabel("n_iterations")
    plt.show()


if __name__ == '__main__':
    experiment_dirs = list(lda_experiments_dir.glob("*"))

    metas = []
    for experiment in experiment_dirs:
        with (experiment / "meta.json").open("r") as f:
            metas.append(json.load(f))

    meta_df = pd.DataFrame(metas)
    print(meta_df.sort_values("test_perplexity"))

    _plot_perplexity_1d(meta_df)
    _plot_perplexity_2d(meta_df)
