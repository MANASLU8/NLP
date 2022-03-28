import os
import json
import glob
from scipy import sparse
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from source.util.file_utils import get_lemmas_list_with_token_tags_from_document_annotation


def get_csr_matrix(docs, tokens_dict, min_fr):
    vocabulary = {}
    indptr = [0]
    indices = []
    data = []
    for d in docs:
        for term in d:
            if tokens_dict[term] >= min_fr:
                index = vocabulary.setdefault(term, len(vocabulary))
                indices.append(index)
                data.append(1)
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr), dtype=int)


def build_csr_term_doc_matrix(annotation_documents_dir_path, min_fr, dict_file, matrix_file):
    tokens_dict = {}
    eng_stopwords = stopwords.words("english")
    file_counter = 1
    docs = []
    # iterate through annotations
    for subdir, dirs, files in os.walk(annotation_documents_dir_path):
        for file in files:
            current_file_path = str(glob.glob(annotation_documents_dir_path + "/**/" + str(file_counter) + ".tsv", recursive=True)[0])
            doc_tokens_list = get_lemmas_list_with_token_tags_from_document_annotation(current_file_path)
            print(file_counter)
            file_counter += 1
            final_tokens_list_for_document = []
            # iterate though tokens from annotation
            for token_info in doc_tokens_list:
                # check for stop words and punctuation
                if (token_info[0].lower() not in eng_stopwords) and (token_info[1] != "punctuation sign") \
                        and (token_info[1] != "quotation") and (token_info[1] != "word with unicode decimal code")\
                        and (token_info[1] != "undefined token type"):
                    token = str(token_info[0].lower())
                    # check if token exist in dict and create or inc value
                    if not (token in tokens_dict):
                        tokens_dict[token] = 0
                    else:
                        tokens_dict[token] += 1
                    # add to final list
                    final_tokens_list_for_document.append(token)
            docs.append(final_tokens_list_for_document)

    # filter resulting dict by frequency
    tokens_dict_to_save = {k: v for k, v in tokens_dict.items() if tokens_dict[k] >= min_fr}
    # building and writing to file dictionary
    f1 = open(dict_file, "w")
    json.dump(tokens_dict_to_save, f1)
    f1.close()
    # getting csr matrix
    td_matrix = get_csr_matrix(docs, tokens_dict, min_fr)
    sparse.save_npz(matrix_file, td_matrix)
