import csv
import json
import logging
import math
import os
import sys
from io import StringIO

from nltk.corpus import stopwords

from .cosine import cosine_distance
from .reduce import pca_reduce

sys.path.append('../text_mining')

from text_mining.annotation import annotate_text


def rle_encode(data_str):
    current_seq = ""
    prev_seq = ""
    encoding = ""
    count = 1
    data = data_str[(str(data_str).find(',')):] + ","
    if not data:
        return ''
    for char in data:
        if char == ',':
            if current_seq == prev_seq:
                count += 1
            else:
                if prev_seq:
                    if count > 1:
                        encoding += str(count) + "_" + prev_seq + ","
                    else:
                        encoding += prev_seq + ","
                count = 1
            prev_seq = current_seq
            current_seq = ""
            continue
        current_seq += char
    if count > 1:
        encoding += str(count) + "_" + prev_seq + ","
    else:
        encoding += prev_seq + ","
    return data_str[:(str(data_str).find(',') + 1)] + encoding[:-1]


def rle_decode(text):
    freq_str = text[(text.find(',') + 1):]
    expanded_text_list = []
    for fr_info in freq_str.split(','):
        if "_" in fr_info:
            amount = int(fr_info.split("_")[0])
            fr = fr_info.split("_")[1]
            for i in range(0, amount):
                expanded_text_list.append(int(fr))
        else:
            expanded_text_list.append(int(fr_info))
    return expanded_text_list


def get_file_lines_number(file):
    file = open(file, "r")
    line_count = 0
    for line in file:
        if line != "\n":
            line_count += 1
    file.close()
    return line_count


def matrix_header_csv_string_to_list(text):
    comm_index = text.find(',') + 1
    line = StringIO(text[comm_index:-1])
    reader = csv.reader(line, delimiter=',')
    split_list = []
    for parts in reader:
        split_list.append(parts)
    return split_list[0]


def get_tokens_list_with_token_tags_from_document_annotation(document_path):
    tokens = []
    with open(document_path, "r") as f:
        line = f.readline()
        while line:
            if line:
                token = line.split("\t")[0]
                if token != '\n':
                    tokens.append((token, line.split("\t")[3][:-1]))
            line = f.readline()
    f.close()
    return tokens


def build_frequency_dict_and_term_doc_matrix(annotation_documents_dir_path, min_fr, dict_file, matrix_file):
    tokens_dict = {}
    eng_stopwords = stopwords.words("english")
    file_counter = 1
    docs_list = []
    for subdir, dirs, files in os.walk(annotation_documents_dir_path):
        for file in files:
            doc_tokens_list = get_tokens_list_with_token_tags_from_document_annotation(os.path.join(subdir, file))
            doc_name = file
            docs_list.append(doc_name)
            logging.info(file_counter)
            file_counter += 1
            for token_info in doc_tokens_list:
                if (token_info[0].lower() not in eng_stopwords) and (token_info[1] != "punctuation sign") \
                        and (token_info[1] != "quotation"):
                    token = str(token_info[0].lower())
                    if not (token in tokens_dict):
                        tokens_dict[token] = {}
                    if doc_name in tokens_dict[token]:
                        tokens_dict[token][doc_name] += 1
                    else:
                        tokens_dict[token][doc_name] = 1
    tokens_dict = {k: v for k, v in tokens_dict.items() if sum(tokens_dict[k].values()) >= min_fr}
    tokens_with_total_frequencies_dict = {}
    for key in tokens_dict:
        tokens_with_total_frequencies_dict[key] = sum(tokens_dict[key].values())
    f1 = open(dict_file, "w")
    json.dump(tokens_with_total_frequencies_dict, f1)
    f1.close()
    f2 = open(matrix_file, "w")
    header_string = ""
    for key in tokens_dict.keys():
        if "," in key:
            header_string += ",\"" + key + "\""
        else:
            header_string += "," + key
    f2.write(header_string)
    f2.write("\n")
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
        logging.info(processed_lines_counter)
        processed_lines_counter += 1
    f2.close()


def read_word_dict_to_dict(dict_file):
    f = open(dict_file, "r")
    fr_dict = json.load(f)
    f.close()
    return fr_dict


def tokenize_with_filter(text):
    eng_stopwords = stopwords.words("english")
    tokens_list = [token_info.token.lower() for token_info in annotate_text(text)
                   if (token_info.token_type != "whitespace" and token_info.token_type != "punctuation"
                       and token_info.token_type != "quotation" and token_info.token.lower() not in eng_stopwords)]
    return tokens_list


def vectorize_custom_text(text, matrix_file):
    tokens_list = tokenize_with_filter(text)
    total_documents_amount = get_file_lines_number(matrix_file) - 2  # two first rows are headers
    f = open(matrix_file, "r")
    dictionary_words_list = matrix_header_csv_string_to_list(f.readline())
    dictionary_words_total_docs_num_row = f.readline()
    dictionary_words_total_docs_num_list = \
        rle_decode(dictionary_words_total_docs_num_row[(dictionary_words_total_docs_num_row.find(',')):-1])
    text_vector = []
    for term in dictionary_words_list:
        tf_value = (tokens_list.count(term)) / float(len(tokens_list))
        idf_value = math.log10(total_documents_amount /
                               int(dictionary_words_total_docs_num_list[dictionary_words_list.index(term)]))
        tf_idf_value = tf_value * idf_value
        text_vector.append(tf_idf_value)
    return text_vector


def get_single_word_tf_idf(word, dict_file, matrix_file):
    words_dict = read_word_dict_to_dict(dict_file)
    total_documents_amount = get_file_lines_number(matrix_file) - 2
    f = open(matrix_file, "r")
    dictionary_words_list = matrix_header_csv_string_to_list(f.readline())
    dictionary_words_total_docs_num_row = f.readline()
    dictionary_words_total_docs_num_list = \
        rle_decode(dictionary_words_total_docs_num_row[(dictionary_words_total_docs_num_row.find(',')):-1])
    if word in words_dict and word in dictionary_words_list:
        tf_value = (words_dict[word]) / float(sum(words_dict.values()))
        idf_value = math.log10(total_documents_amount /
                               int(dictionary_words_total_docs_num_list[dictionary_words_list.index(word)]))
        tf_idf_value = tf_value * idf_value
        return tf_idf_value
    else:
        return None


def use_tdidf_model(matrix_file, token_to_test, similar_tokens, same_field_tokens, other_tokens):
    logging.info("Custom tf-idf vectorization")
    all_tokens = []
    all_tokens.append(token_to_test)
    all_tokens = all_tokens + similar_tokens + same_field_tokens + other_tokens
    x = [vectorize_custom_text(token, matrix_file) for token in all_tokens]
    reduced_embeddings = pca_reduce(x, 2)
    logging.info("Cosine distance " + all_tokens[0])
    for i in range(1, len(all_tokens)):
        logging.info(
            "\twith " + all_tokens[i] + ": " + str(cosine_distance(reduced_embeddings[0], reduced_embeddings[i])))
    return reduced_embeddings, all_tokens
