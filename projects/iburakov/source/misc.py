import logging

import pandas as pd


def configure_logging():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def configure_dataframes_printing():
    pd.set_option('display.max_rows', 2000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('expand_frame_repr', False)
