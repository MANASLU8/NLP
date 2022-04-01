from collections import Counter

import pandas as pd
from pandas import DataFrame

from paths import annotated_corpus_dir, token_dictionary_filepath
from task1.utils import read_tokens_from_annotated_corpus_tsv
from task2.spell_correction import CORRECTABLE_TOKEN_TAGS

TOKEN_COUNT_THRESHOLD = 3


def _describe_token_df(df: DataFrame):
    print(f"{df.index.size} unique token(s), {df.counts.sum()} in total, {df.counts.mean():5f} per token on average.")
    print("=== Token Tags ===")
    print(df.value_counts(subset="tag"))
    print("=== Top Tokens ===")
    print(df.sort_values("counts", ascending=False).head(50))


if __name__ == '__main__':
    files = list(annotated_corpus_dir.glob("train/*/*"))
    print(f"Files found: {len(files)}")

    counter = Counter()

    for i, filepath in enumerate(files):
        log_prefix = f"({i + 1}/{len(files)})"
        print(f"{log_prefix} Processing {filepath}")

        tokens = read_tokens_from_annotated_corpus_tsv(filepath)
        final_tokens = []
        for token, tag in tokens.values:
            # lowering WORDs - disabled
            # final_tokens.append((token, tag) if tag != TokenTag.WORD else (token.lower(), tag))
            final_tokens.append((token, tag))

        counter.update(final_tokens)

    df = pd.DataFrame(({"token": k[0], "tag": k[1], "counts": v} for k, v in counter.items())).set_index("token")
    print("Done!")
    _describe_token_df(df)
    print(f"Filtering. Tags: {CORRECTABLE_TOKEN_TAGS}, count threshold: {TOKEN_COUNT_THRESHOLD}")
    df = df[df.tag.isin(CORRECTABLE_TOKEN_TAGS) & (df.counts >= TOKEN_COUNT_THRESHOLD)]
    print("Done!")
    _describe_token_df(df)

    print(f"Writing tokens to {token_dictionary_filepath}.")
    df.sort_values("counts", ascending=False).to_csv(token_dictionary_filepath, sep="\t", line_terminator="\n")
