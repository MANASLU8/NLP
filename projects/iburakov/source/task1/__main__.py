from random import shuffle
from time import perf_counter

from dirs import dataset_dir, annotated_corpus_dir
from task1.lemmatizer import add_lemmas_to_tokens_dataframe
from task1.newsgroup_message import read_newsgroup_message
from task1.stemmer import add_stems_to_tokens_dataframe
from task1.token_tag import TokenTag
from task1.tokenizer import tokenize_text

message_files = list(dataset_dir.glob("*/*/*"))
shuffle(message_files)
print(f"Files to process found: {len(message_files)}")

for i, filepath in enumerate(message_files):
    *_, dataset_dir, category, doc_id = filepath.parts
    dataset = "test" if dataset_dir.endswith("test") else "train"
    target_filename_dir = annotated_corpus_dir / dataset / category
    target_filename = target_filename_dir / f"{doc_id}.tsv"

    log_prefix = f"({i + 1}/{len(message_files)})"
    if target_filename.exists():
        print(f"{log_prefix} Skipping {filepath}, already exists")
        continue
    else:
        print(f"{log_prefix} Processing {filepath}")
        target_filename_dir.mkdir(parents=True, exist_ok=True)

    try:
        start_time = perf_counter()
        print(f"Reading file...", end=" ")
        msg = read_newsgroup_message(filepath)
        print(f"Tokenizing text...", end=" ")
        tokens = tokenize_text(msg.body)
        assert len(tokens), "No tokens found! Empty text?"
        print(f"Stemming...", end=" ")
        add_stems_to_tokens_dataframe(tokens)
        print(f"Lemmatizing...")
        add_lemmas_to_tokens_dataframe(tokens)

        print(f"Writing result to {target_filename}...")
        tsv_columns_order = ["token", "stem", "lemma", "tag"]
        tsv = tokens[tsv_columns_order].to_csv(sep="\t", index=False, header=False, line_terminator="\n")
        # split sentences with extra line break
        tsv = tsv.replace(TokenTag.PUNCT_SENTENCE, TokenTag.PUNCT_SENTENCE + "\n")
        with target_filename.open("w", encoding="utf8") as f:
            f.write(tsv)

        print(f"Done in {(perf_counter() - start_time) * 1000:.3f} ms!")
    except Exception as e:
        print(f"Failed with {e.__class__.__name__}: {e}.")
