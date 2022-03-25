from pathlib import Path

dataset_dir = Path() / "assets" / "raw-dataset"
annotated_corpus_dir = Path() / "assets" / "annotated-corpus"
corrupted_dataset_dir = Path() / "assets" / "corrupted-dataset"
misc_dir = Path() / "assets" / "misc"

token_dictionary_filepath = annotated_corpus_dir / "tokens.tsv"
