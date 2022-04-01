import logging
from itertools import product

from misc import configure_logging, configure_dataframes_printing
from task4.lda_training_evaluation import show_experiment_results, train_evaluate_lda

if __name__ == '__main__':
    configure_logging()
    configure_dataframes_printing()

    n_topics = [2, 5, 10, 20, 40]
    n_iterations = [5, 10, 20]
    search_space = list(product(n_topics, n_iterations))
    for i, (n_topics, n_iterations) in enumerate(search_space):
        logging.info(f"({i + 1}/{len(search_space)}) Doing {(n_topics, n_iterations)}...")
        lda, data_dir = train_evaluate_lda(n_topics, n_iterations)
        show_experiment_results(data_dir)
