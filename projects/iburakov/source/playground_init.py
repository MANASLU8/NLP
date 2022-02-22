from random import choice

import pandas as pd

from dirs import dataset_dir
from task1.newsgroup_message import read_newsgroup_message
from task1.stemmer import get_word_stem, add_stems_to_tokens_dataframe
from task1.tokenizer import tokenize_text

pd.set_option("display.max_rows", 2000)

# noinspection PyStatementEffect
tokenize_text, get_word_stem, add_stems_to_tokens_dataframe

msg_paths = list(dataset_dir.glob("*/*/*"))


def tokenize_random():
    msg = read_newsgroup_message(choice(msg_paths))
    print(msg.body)
    print('---')
    return tokenize_text(msg.body)
