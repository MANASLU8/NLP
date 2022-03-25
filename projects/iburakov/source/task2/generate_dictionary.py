from collections import Counter

import pandas as pd
from pandas import DataFrame

from dirs import annotated_corpus_dir
from task1.token_tag import TokenTag

CORRECTABLE_TOKEN_TAGS = {TokenTag.PGP_BEGINNING, TokenTag.WORD, TokenTag.ABBREVIATION, TokenTag.PERSON_NAME}
TOKEN_COUNT_THRESHOLD = 10


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

        tokens = pd.read_csv(filepath, sep="\t", header=None, keep_default_na=False)
        tokens.columns = ["token", "stem", "lemma", "tag"]

        final_tokens = []
        for token, tag in tokens[["token", "tag"]].values:
            # lowering WORDs - disabled
            # final_tokens.append((token, tag) if tag != TokenTag.WORD else (token.lower(), tag))
            final_tokens.append((token, tag))

        counter.update(final_tokens)

    df = pd.DataFrame(({"token": k[0], "tag": k[1], "counts": v} for k, v in counter.items())).set_index("token")
    print("Done!")
    _describe_token_df(df)
    print(f"Filtering. Tags: {CORRECTABLE_TOKEN_TAGS}, count threshold: {TOKEN_COUNT_THRESHOLD}")
    df = df[df.tag.isin(CORRECTABLE_TOKEN_TAGS) & (df.counts > TOKEN_COUNT_THRESHOLD)]
    print("Done!")
    _describe_token_df(df)

    output_filename = annotated_corpus_dir / "tokens.tsv"
    print(f"Writing tokens to {output_filename}.")
    df.sort_values("counts", ascending=False).to_csv(output_filename, sep="\t", line_terminator="\n")
