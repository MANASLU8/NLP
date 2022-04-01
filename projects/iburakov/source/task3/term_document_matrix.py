from itertools import repeat
from pathlib import Path

from scipy import sparse
from scipy.sparse import load_npz

from dirs import term_document_matrix_filepath
from task1.utils import read_tokens_from_annotated_corpus_tsv
from task2.spell_correction import CORRECTABLE_TOKEN_TAGS
from task3.token_dictionary import load_token_dictionary

_dct = load_token_dictionary()


def generate_term_document_matrix(target_filepath: Path, source_dir: Path):
    files = list(source_dir.glob("*/*"))
    print(f"Files found: {len(files)}")

    row_ind_lists = []
    col_ind_lists = []
    data_lists = []

    for i, filepath in enumerate(files):
        log_prefix = f"({i + 1}/{len(files)})"
        print(f"{log_prefix} Processing {filepath}")

        tokens = read_tokens_from_annotated_corpus_tsv(filepath)
        tokens = tokens[tokens.tag.isin(CORRECTABLE_TOKEN_TAGS)].token
        counts = tokens[tokens.isin(_dct.tokens)].value_counts(sort=False)
        token_indices = _dct.df.token.reset_index().set_index("token").reindex(counts.index)["index"].values

        row_ind_lists.extend(repeat(i, len(token_indices)))  # i - index of document, goes to row
        col_ind_lists.extend(token_indices)  # token indices go to columns
        data_lists.extend(counts.values)

    s = sparse.coo_matrix((data_lists, (row_ind_lists, col_ind_lists)),
                          shape=(len(files), len(_dct.tokens)), dtype="int32")
    sparse.save_npz(target_filepath, s, compressed=True)


def load_term_document_matrix():
    return load_npz(term_document_matrix_filepath)
