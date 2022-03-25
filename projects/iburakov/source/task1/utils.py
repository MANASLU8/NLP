from typing import Pattern, AnyStr, Iterable, Tuple, Callable

import pandas as pd
from pandas import DataFrame

from task1.token_tag import TokenTag


def split_keep_delimiter_and_metadata(delimiter_re: Pattern[AnyStr], s: str) -> Iterable[Tuple[bool, str]]:
    """
    :returns [(is_delimiter: bool, split_part: str)]
    """
    last_end = 0
    for match in delimiter_re.finditer(s):
        start, end = match.span()

        # yield everything before match if there's something
        if last_end != start:
            yield False, s[last_end:start]

        # yield the match itself
        if start != end:
            yield True, match.group()

        last_end = end

    # yield what's left after all matches if there's something
    if last_end != len(s):
        yield False, s[last_end:]


def add_mapped_words_to_tokens_dataframe(df: DataFrame, new_column_name: str, word_mapper: Callable[[str], str]):
    df[new_column_name] = pd.NA
    df.loc[df.tag == TokenTag.WORD, new_column_name] = df.loc[df.tag == TokenTag.WORD].token.apply(word_mapper)
    df.loc[pd.isna(df[new_column_name]), new_column_name] = df.loc[pd.isna(df[new_column_name])].token


def read_tokens_from_annotated_corpus_tsv(filepath_or_buf) -> DataFrame:
    tokens = pd.read_csv(filepath_or_buf, sep="\t", header=None, keep_default_na=False)
    tokens.columns = ["token", "stem", "lemma", "tag"]
    return tokens[["token", "tag"]]
