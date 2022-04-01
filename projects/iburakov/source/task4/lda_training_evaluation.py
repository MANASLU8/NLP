import json
import logging
from pathlib import Path
from shutil import rmtree

import pandas as pd
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation

from paths import lda_experiments_dir, test_term_document_matrix_filepath, annotated_corpus_dir
from task3.term_document_matrix import load_term_document_matrix, get_term_document_matrix_docs
from task3.token_dictionary import load_token_dictionary

_dct = load_token_dictionary()


def train_evaluate_lda(n_topics: int, n_iterations: int):
    lda = LatentDirichletAllocation(n_topics, max_iter=n_iterations, n_jobs=-2)
    logging.info("Loading TDM")
    tdm = load_term_document_matrix()
    logging.info(f"Done, shape: {tdm.shape}")
    logging.info(f"Fitting {lda}")
    lda.fit(tdm)
    data_path = _generate_experiment_data_dir(lda)
    return lda, data_path


def show_experiment_results(data_dir: Path):
    with (data_dir / "meta.json").open("r") as f:
        meta = json.load(f)

    print(f"Experiment {meta}")
    print("Top words:")
    print(pd.read_csv(data_dir / "top-words-per-topic.tsv", sep="\t", header=None, index_col=0).T)
    print("Top documents:")
    print(pd.read_csv(data_dir / "top-documents-per-topic.tsv", sep="\t", header=None, index_col=0).T)


def get_experiment_data_path(n_topics, n_iterations):
    return lda_experiments_dir / f"topics-{n_topics} iters-{n_iterations}"


def _generate_experiment_data_dir(lda: LatentDirichletAllocation, n_top_words=30, n_top_docs=20):
    n_topics = lda.n_components
    n_iterations = lda.max_iter
    target_dir = get_experiment_data_path(n_topics, n_iterations)

    logging.info(f"Generating experiment data to {target_dir}")
    if target_dir.exists():
        logging.info(f"Found old data, overwriting")
        rmtree(target_dir)
    target_dir.mkdir(parents=True)

    # commented since not needed + weights a lot
    # logging.info(f"Dumping model")
    # with (target_dir / "model.pickle").open("wb") as f:
    #     pickle.dump(lda, f)

    logging.info(f"Generating {n_top_words} top words per topic matrix")
    pd.DataFrame({
        i: [_dct.tokens_list[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        for i, topic in enumerate(lda.components_)
    }).T.to_csv(target_dir / "top-words-per-topic.tsv", sep="\t", line_terminator="\n", header=False)

    ttdm = load_term_document_matrix(test_term_document_matrix_filepath)
    logging.info(f"Evaluating model's test perplexity on test term document matrix, shape {ttdm.shape}")
    test_perplexity = lda.perplexity(ttdm)
    logging.info(f"Done! Test perplexity: {test_perplexity}")

    meta_fname = "meta.json"
    logging.info(f"Writing {meta_fname}")
    with (target_dir / meta_fname).open("w") as f:
        json.dump({
            "test_perplexity": test_perplexity,
            "n_topics": n_topics,
            "n_iterations": n_iterations,
        }, f)

    doc_topics = _get_documents_topics_test_dataset_df(lda, ttdm)
    doc_topics_fname = "document-topics.tsv"
    logging.info(f"Writing {doc_topics_fname}")
    doc_topics.to_csv(target_dir / doc_topics_fname, sep="\t", line_terminator="\n", header=False)

    logging.info(f"Generating {n_top_docs} top docs per topic matrix")
    pd.DataFrame({
        topic: doc_topics[topic].sort_values(ascending=False)[:n_top_docs].index
        for topic in doc_topics.columns
    }).T.to_csv(target_dir / "top-documents-per-topic.tsv", sep="\t", line_terminator="\n", header=False)

    logging.info(f"Done generating experiment data to {target_dir}")
    return target_dir


def _get_documents_topics_test_dataset_df(lda: LatentDirichletAllocation, ttdm: sparse.coo_matrix):
    _, names = get_term_document_matrix_docs(annotated_corpus_dir / "test")
    logging.info(f"Generating Document-TopicsVector table for test dataset, {len(names)} files, "
                 f"TDM matrix {ttdm.shape}")
    doc_topic_vecs = lda.transform(ttdm)
    return pd.DataFrame(doc_topic_vecs, index=names)
