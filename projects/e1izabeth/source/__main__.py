import os
from os import listdir
from pathlib import Path

import pandas as pd
import regex

from tokenizer.tokenizer.tokenizer import tokenize_text

classes = dict([
    (1, 'World'),
    (2, 'Sports'),
    (3, 'Business'),
    (4, 'Sci-Tech')
])


def main():
    print("main")


if __name__ == "__main__":
    main()
