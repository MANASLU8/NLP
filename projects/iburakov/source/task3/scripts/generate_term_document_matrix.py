from itertools import repeat

from scipy import sparse

from dirs import term_document_matrix_filepath
from task1.utils import read_tokens_from_annotated_corpus_tsv
from task2.spell_correction import CORRECTABLE_TOKEN_TAGS
from task3.term_document_matrix import get_term_document_matrix_documents
from task3.token_dictionary import load_token_dictionary

if __name__ == '__main__':
    files, docs = get_term_document_matrix_documents()
    # files = files[:50]
    print(f"Files found: {len(files)}")

    dct = load_token_dictionary()

    row_ind_lists = []
    col_ind_lists = []
    data_lists = []

    for i, filepath in enumerate(files):
        log_prefix = f"({i + 1}/{len(files)})"
        print(f"{log_prefix} Processing {filepath}")

        tokens = read_tokens_from_annotated_corpus_tsv(filepath)
        tokens = tokens[tokens.tag.isin(CORRECTABLE_TOKEN_TAGS)].token
        counts = tokens[tokens.isin(dct.tokens)].value_counts(sort=False)
        token_indices = dct.df.token.reset_index().set_index("token").reindex(counts.index)["index"].values

        row_ind_lists.extend(repeat(i, len(token_indices)))  # i - index of document, goes to row
        col_ind_lists.extend(token_indices)  # token indices go to columns
        data_lists.extend(counts.values)

    s = sparse.coo_matrix((data_lists, (row_ind_lists, col_ind_lists)),
                          shape=(len(docs), len(dct.tokens)), dtype="int32")
    sparse.save_npz(term_document_matrix_filepath, s, compressed=True)
