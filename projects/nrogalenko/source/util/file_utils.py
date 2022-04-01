import csv
from io import StringIO


def get_tokens_list_from_document_annotation(document_path):
    tokens = []
    with open(document_path, "r") as f:
        line = f.readline()
        while line:
            if line:
                token = line.split("    ")[0]
                if token != '\n':
                    tokens.append(token)
            line = f.readline()
    f.close()
    return tokens


def get_tokens_list_with_token_tags_from_document_annotation(document_path):
    tokens = []
    with open(document_path, "r") as f:
        line = f.readline()
        while line:
            if line:
                token = line.split("    ")[0]
                if token != '\n':
                    tokens.append((token, line.split("    ")[4][:-1]))
            line = f.readline()
    f.close()
    return tokens


def get_lemmas_list_with_token_tags_from_document_annotation(document_path):
    tokens = []
    with open(document_path, "r") as f:
        line = f.readline()
        while line:
            if line:
                token = line.split("    ")[0]
                if token != '\n':
                    tokens.append((line.split("    ")[2], line.split("    ")[4][:-1]))
            line = f.readline()
    f.close()
    return tokens


def get_document_annotation_tokens_count(path):
    counter = 0
    with open(path, "r") as f:
        line = f.readline()
        while line:
            if line:
                counter += 1
            line = f.readline()
    f.close()
    return counter


def matrix_header_csv_string_to_list(text):
    comm_index = text.find(',') + 1
    line = StringIO(text[comm_index:-1])
    reader = csv.reader(line, delimiter=',')
    split_list = []
    for parts in reader:
        split_list.append(parts)
    return split_list[0]


def get_file_lines_number(file):
    file = open(file, "r")
    line_count = 0
    for line in file:
        if line != "\n":
            line_count += 1
    file.close()
    return line_count
