import json
from pathlib import Path

from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation
import joblib


def get_train_line(path, line_num):
    train_path = Path(str(Path(Path.cwd()))[:-len("source")], "assets", "resources", path)
    with open(train_path, "r", encoding="utf-8") as fin:
        for line_count, line in enumerate(fin):
            if line_count == line_num:
                return line


def lda_train(topics_count, max_iter, csr_path):
    csr_matrix_file = Path(str(Path(Path.cwd()))[:-len("source")], "assets", "lda", csr_path)
    td_matrix = sparse.load_npz(csr_matrix_file)
    lda_model = LatentDirichletAllocation(n_components=topics_count, max_iter=max_iter, learning_method='online') \
        .fit(td_matrix)
    joblib.dump(lda_model, str(Path(str(Path(Path.cwd()))[:-len("source")], "assets", "lda", "models", str(max_iter)
                                    + '_' + str(topics_count) + '.jl')))


def lda_fit(mod_path, fdict_path, csr_path, train_path, path, top_number_to_show, topdoc_path):
    train_topdoc_path = Path(str(Path(Path.cwd()))[:-len("source")], "assets", "lda", path)
    # загружаем модель
    lda = joblib.load(mod_path)
    # берем словарь частот из файла
    with open(fdict_path, "r", encoding="utf-8") as file:
        fdict = json.load(file)
    # все слова
    words = list(fdict.keys())
    # получаем распределение по темам документа
    csr_matrix = sparse.load_npz(csr_path)
    doc_topic = lda.transform(csr_matrix)

    # записываем вероятности документ-тема
    docs_probs4topic = [[] for _ in range(len(doc_topic[0]))]
    with open(Path(str(Path(Path.cwd()))[:-len("source")], "assets", "lda",
                   train_topdoc_path), "w+") as fout:
        for c, doc in enumerate(doc_topic):
            doc_topic_str = str(c + 1)
            for prob_count, probability in enumerate(doc):
                docs_probs4topic[prob_count].append(probability)
                doc_topic_str += ("\t" + str(probability))
            fout.write(doc_topic_str)
            fout.write("\n")

    # записываем топ ключевых слов
    with open(Path(str(Path(Path.cwd()))[:-len("source")], "assets", "lda",
                   "top10_" + path), "w+", encoding="utf-8") as fout:
        for topic_count, topic in enumerate(lda.components_):
            fout.write("Тема: "+str(topic_count + 1))
            fout.write("\n")
            fout.write(" ".join([words[x] for x in topic.argsort()[:-top_number_to_show - 1:-1]]))
            fout.write("\n")
            top_documents_indexes = sorted(range(len(docs_probs4topic[topic_count])),
                                           key=lambda i: docs_probs4topic[topic_count][i])[-top_number_to_show:]
            for i in top_documents_indexes:
                fout.write(get_train_line(train_path, i))
            fout.write("\n")

    # записываем perplexity
    with open(Path(str(Path(Path.cwd()))[:-len("source")], "assets", "lda",
                   "perplexity.tsv"), "a+", encoding="utf-8") as fout:
        topdoc_matrix = sparse.load_npz(topdoc_path)
        fout.write(str(lda.max_iter) + "," + str(lda.n_components) + ","
                   + str(lda.perplexity(topdoc_matrix)) + "\n")
