from pathlib import Path

# TODO: rename to "paths"

dataset_dir = Path() / "assets" / "raw-dataset"
annotated_corpus_dir = Path() / "assets" / "annotated-corpus"
corrupted_dataset_dir = Path() / "assets" / "corrupted-dataset"
misc_dir = Path() / "assets" / "misc"

token_dictionary_filepath = annotated_corpus_dir / "tokens.tsv"
term_document_matrix_filepath = annotated_corpus_dir / "term-document-matrix.npz"
test_embeddings_filepath = annotated_corpus_dir / "test-embeddings.tsv"
word2vec_model_filepath = misc_dir / "word2vec.model"
