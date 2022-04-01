from pathlib import Path

from lab4.plot import build_plot
from lab4.make_csr import make_csr
from lab4.lda import lda_train, lda_fit


def lab4(train_path, csr_path, train_topdoc_path, fdict_path, test_path, test_topdoc_path):
    # PART 1
    # топ-10 ключевых слов для каждой темы;
    # perplexity полученной модели на тестовой выборке
    # вероятность принадлежности документов обучающей выборки к той или иной теме

    make_csr(train_path, csr_path, 5)
    print("train csr done")
    make_csr(test_path, test_topdoc_path, 5)
    print("test csr done")
    iterations = [10, 20, 40]
    topics = [4]
    top_words_to_display_num = 10
    for i in iterations:
        print("количество итераций: ", i)
        for topic in topics:
            print("количество тем: ", topic)
            lda_train(topic, i, csr_path)
            lda_fit(str(Path(str(Path(Path.cwd()))[:-len("source")], "assets", "lda", "models", str(i) +
                             '_' + str(topic) + '.jl')), fdict_path, csr_path, train_path,
                    train_topdoc_path, top_words_to_display_num, test_topdoc_path)
    # PART 2 см. plot.py
    # график изменения perplexity в зависимости от количества тем
    build_plot(Path(str(Path(Path.cwd()))[:-len("source")], "assets", "lda", "perplexity.tsv"))
