from scipy.sparse import load_npz

from dirs import term_document_matrix_filepath, annotated_corpus_dir


def get_term_document_matrix_documents():
    paths = list(annotated_corpus_dir.glob("train/*/*"))
    names = [f.name[:-4] for f in paths]
    return paths, names


def load_term_document_matrix():
    _, docs = get_term_document_matrix_documents()
    return load_npz(term_document_matrix_filepath)
