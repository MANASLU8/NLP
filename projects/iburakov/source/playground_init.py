from random import choice

from misc import configure_dataframes_printing, configure_logging
from paths import dataset_dir, test_term_document_matrix_filepath
from task1.newsgroup_message import read_newsgroup_message
from task1.stemmer import get_word_stem, add_stems_to_tokens_dataframe
from task1.tokenizer import tokenize_text
from task2.qwerty_weighting import get_qwerty_weighted_substitution_cost
from task2.sequence_alignment import get_alignment_and_wagner_fischer_matrix, get_optimal_alignment_hirschberg, \
    alignment_to_strings, get_edit_distance, evaluate_alignment_edit_distance
from task2.spell_correction import _get_token_correction
from task3.term_document_matrix import load_term_document_matrix
from task3.token_dictionary import load_token_dictionary
from task3.vectorizing_simple import vectorize_text_simple
from task3.vectorizing_w2v import vectorize_text_w2v
from task4.lda_training_evaluation import train_evaluate_lda, _get_documents_topics_test_dataset_df, \
    show_experiment_results, get_experiment_data_path

configure_dataframes_printing()
configure_logging()

# noinspection PyStatementEffect
(tokenize_text, get_word_stem, add_stems_to_tokens_dataframe, get_qwerty_weighted_substitution_cost,
 get_alignment_and_wagner_fischer_matrix, get_optimal_alignment_hirschberg, alignment_to_strings, get_edit_distance,
 evaluate_alignment_edit_distance, _get_token_correction, vectorize_text_simple, vectorize_text_w2v, train_evaluate_lda,
 _get_documents_topics_test_dataset_df,)

msg_paths = list(dataset_dir.glob("*/*/*"))


def tokenize_random():
    msg = read_newsgroup_message(choice(msg_paths))
    print(msg.body)
    print('---')
    return tokenize_text(msg.body)


dct = load_token_dictionary()

ttdm = load_term_document_matrix(test_term_document_matrix_filepath)


def show_lda_experiment(n_topics=20, n_iterations=10):
    show_experiment_results(get_experiment_data_path(n_topics, n_iterations))
