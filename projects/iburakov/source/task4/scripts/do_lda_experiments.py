import logging
from itertools import product

from misc import configure_logging, configure_dataframes_printing
from task4.lda_training_evaluation import show_experiment_results, train_evaluate_lda, get_experiment_data_path

if __name__ == '__main__':
    configure_logging()
    configure_dataframes_printing()

    n_iterations = [1, 2, 3, 5, 10, 20, 30, 40]
    n_topics = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 30, 40]
    search_space = list(product(n_topics, n_iterations))

    for i, (n_topics, n_iterations) in enumerate(search_space):
        logging_prefix = f"({i + 1}/{len(search_space)})"

        data_dir = get_experiment_data_path(n_topics, n_iterations)
        if data_dir.exists():
            logging.info(f"{logging_prefix} Skipping {(n_topics, n_iterations)} - exists...")
            continue

        logging.info(f"{logging_prefix} Doing {(n_topics, n_iterations)}...")
        lda, _ = train_evaluate_lda(n_topics, n_iterations)
        show_experiment_results(data_dir)
