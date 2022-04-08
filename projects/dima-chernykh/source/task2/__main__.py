import glob
import os
import json
from pathlib import Path

import pandas as pd
from pandas.core.dtypes.common import classes

from source.task1.tokenizer import tokenize
from source.task2.corrupted_file_handler import handle_corrupted_file
from source.task2.file_helper import create_dict, get_dict, get_tokens_from_annotated_corpus


# create_dict('assets/annotated-corpus/test', 'assets/dict.tsv')
handle_corrupted_file('assets/annotated-corpus-corrupted', 'assets/annotated-corpus', 'assets/dict.tsv')
