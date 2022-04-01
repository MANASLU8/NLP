from pathlib import Path

# TODO: rename to "paths"

assets_dir = Path() / "assets"
dataset_dir = assets_dir / "raw-dataset"
annotated_corpus_dir = assets_dir / "annotated-corpus"
corrupted_dataset_dir = assets_dir / "corrupted-dataset"
misc_dir = assets_dir / "misc"
lda_experiments_dir = assets_dir / "lda_experiments"

token_dictionary_filepath = annotated_corpus_dir / "tokens.tsv"
term_document_matrix_filepath = annotated_corpus_dir / "term-document-matrix.npz"
test_term_document_matrix_filepath = annotated_corpus_dir / "test-term-document-matrix.npz"
test_embeddings_filepath = annotated_corpus_dir / "test-embeddings.tsv"
word2vec_model_filepath = misc_dir / "word2vec.model"
