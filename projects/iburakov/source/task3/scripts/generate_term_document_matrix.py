from paths import term_document_matrix_filepath, annotated_corpus_dir
from task3.term_document_matrix import generate_term_document_matrix

if __name__ == '__main__':
    generate_term_document_matrix(term_document_matrix_filepath, annotated_corpus_dir / "train")
