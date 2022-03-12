import os
import json
import math
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from source.util.file_utils import get_tokens_list_with_token_tags_from_document_annotation,\
    matrix_header_csv_string_to_list, get_file_lines_number
from source.util.rle_encoder import rle_encode, rle_decode
from source.text_annotation import tokenizer


def build_frequency_dict_and_term_doc_matrix(annotation_documents_dir_path, min_fr, dict_file, matrix_file):
    tokens_dict = {}
    eng_stopwords = stopwords.words("english")
    file_counter = 1
    docs_list = []
    # iterate through annotations
    for subdir, dirs, files in os.walk(annotation_documents_dir_path):
        for file in files:
            doc_tokens_list = get_tokens_list_with_token_tags_from_document_annotation(os.path.join(subdir, file))
            doc_name = file
            docs_list.append(doc_name)
            print(file_counter)
            file_counter += 1
            # iterate though tokens from annotation
            for token_info in doc_tokens_list:
                # check for stop words and punctuation
                if (token_info[0].lower() not in eng_stopwords) and (token_info[1] != "punctuation sign") \
                        and (token_info[1] != "quotation"):
                    token = str(token_info[0].lower())
                    # check if token exist in dict
                    if not (token in tokens_dict):
                        # create dict of docs for token
                        tokens_dict[token] = {}
                    # check if document for this token exist in dict
                    if doc_name in tokens_dict[token]:
                        tokens_dict[token][doc_name] += 1
                    else:
                        tokens_dict[token][doc_name] = 1
    # filter resulting dict by frequency
    tokens_dict = {k: v for k, v in tokens_dict.items() if sum(tokens_dict[k].values()) >= min_fr}
    # building and writing to file dictionary
    tokens_with_total_frequencies_dict = {}
    for key in tokens_dict:
        tokens_with_total_frequencies_dict[key] = sum(tokens_dict[key].values())
    # print(tokens_with_total_frequencies_dict)
    f1 = open(dict_file, "w")
    json.dump(tokens_with_total_frequencies_dict, f1)
    f1.close()
    # building and writing to file term document matrix
    f2 = open(matrix_file, "w")
    header_string = ""
    for key in tokens_dict.keys():
        if "," in key:
            header_string += ",\"" + key + "\""
        else:
            header_string += "," + key
    f2.write(header_string)
    f2.write("\n")
    # add total docs containing token number
    f2.write('total_docs,' + rle_encode(','.join([str(len(tokens_dict[k])) for k in tokens_dict])))
    f2.write("\n")
    processed_lines_counter = 1
    for doc in docs_list:
        freq_string = doc
        for key in tokens_dict:
            if doc in tokens_dict[key]:
                freq_string += "," + str(tokens_dict[key][doc])
            else:
                freq_string += ",0"
        f2.write(rle_encode(freq_string))
        f2.write("\n")
        print(processed_lines_counter)
        processed_lines_counter += 1
    f2.close()


def read_word_dict_to_dict(dict_file):
    f = open(dict_file, "r")
    fr_dict = json.load(f)
    f.close()
    return fr_dict


# remove stopwords, punctuation and so on
def tokenize_with_filter(text):
    eng_stopwords = stopwords.words("english")
    # filter tokens
    tokens_list = [token_info.token.lower() for token_info in tokenizer.tokenize(text)
                   if (token_info.token_tag != "whitespace" and token_info.token_tag != "punctuation sign"
                       and token_info.token_tag != "quotation" and token_info.token.lower() not in eng_stopwords)]
    return tokens_list


def reduce_text_vector(vector, dim):
    pca = PCA(n_components=dim)
    result = pca.transform(vector)
    return result


# using tf-idf
def vectorize_custom_text(text, fr_dict, matrix_file):
    tokens_list = tokenize_with_filter(text)
    total_documents_amount = get_file_lines_number(matrix_file) - 2  # two first rows are headers
    f = open(matrix_file, "r")
    dictionary_words_list = matrix_header_csv_string_to_list(f.readline())
    # print(dictionary_words_list)
    dictionary_words_total_docs_num_row = f.readline()
    dictionary_words_total_docs_num_list = rle_decode(dictionary_words_total_docs_num_row[(text.find(',') + 1):-1])
    text_vector = []
    # print(dictionary_words_list)
    # print(dictionary_words_total_docs_num_list)
    for term in dictionary_words_list:
        tf_value = (tokens_list.count(term)) / float(len(tokens_list))
        idf_value = math.log10(total_documents_amount /
                               int(dictionary_words_total_docs_num_list[dictionary_words_list.index(term)]))
        tf_idf_value = tf_value * idf_value
        text_vector.append(round(tf_idf_value, 3))
    # print(text_vector)
    return text_vector


def process_text_for_vectorization(text, dict_file, matrix_file):
    words_dict = read_word_dict_to_dict(dict_file)
    return vectorize_custom_text(text, words_dict, matrix_file)
