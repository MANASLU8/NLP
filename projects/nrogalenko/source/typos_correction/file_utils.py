import os


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
