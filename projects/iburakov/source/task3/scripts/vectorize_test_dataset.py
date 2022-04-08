import pandas as pd

from paths import annotated_corpus_dir, test_embeddings_filepath
from task1.utils import read_tokens_from_annotated_corpus_tsv
from task3.vectorizing_w2v import vectorize_text_w2v

if __name__ == '__main__':
    files = list(annotated_corpus_dir.glob("test/*/*"))
    print(f"Files found: {len(files)}")

    df = pd.DataFrame(columns=range(100))

    for i, filepath in enumerate(files):
        log_prefix = f"({i + 1}/{len(files)})"
        print(f"{log_prefix} Processing {filepath}")

        try:
            tokens = read_tokens_from_annotated_corpus_tsv(filepath)
            result = vectorize_text_w2v(tokens)
            df.loc[filepath.name[:-4], :] = result
        except Exception as e:
            print(f"Exception for {filepath}, {e.__class__.__name__}: {e}")
            continue

    print(f"Writing to {test_embeddings_filepath}...")
    df.to_csv(test_embeddings_filepath, sep="\t", line_terminator="\n", header=False)
