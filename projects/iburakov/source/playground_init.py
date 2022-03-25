from random import choice

import pandas as pd

from dirs import dataset_dir
from task1.newsgroup_message import read_newsgroup_message
from task1.stemmer import get_word_stem, add_stems_to_tokens_dataframe
from task1.tokenizer import tokenize_text
from task2.qwerty_weighting import get_qwerty_weighted_substitution_cost
from task2.sequence_alignment import get_alignment_and_wagner_fischer_matrix, get_optimal_alignment_hirschberg, \
    alignment_to_strings, get_edit_distance, evaluate_alignment_edit_distance
from task2.spell_correction import _get_token_correction

pd.set_option("display.max_rows", 2000)

# noinspection PyStatementEffect
(tokenize_text, get_word_stem, add_stems_to_tokens_dataframe, get_qwerty_weighted_substitution_cost,
 get_alignment_and_wagner_fischer_matrix, get_optimal_alignment_hirschberg, alignment_to_strings, get_edit_distance,
 evaluate_alignment_edit_distance, _get_token_correction)

msg_paths = list(dataset_dir.glob("*/*/*"))

_get_token_correction("msk")


def tokenize_random():
    msg = read_newsgroup_message(choice(msg_paths))
    print(msg.body)
    print('---')
    return tokenize_text(msg.body)
