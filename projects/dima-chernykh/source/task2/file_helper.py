import csv
import os

import pandas as pd


def create_dict(token_files_dir_path, resulting_dictionary_path):
    os.makedirs(os.path.dirname("/".join((resulting_dictionary_path + "/").split("/")[:-1])), exist_ok=True)
    with open(resulting_dictionary_path, "w") as f:
        for root, dirs, files in os.walk(token_files_dir_path):
            for file in files:
                df = pd.read_csv(root + "/" + file, delimiter="\t", header=None, usecols=[0], engine='python')
                for item in df.iloc[:, 0].tolist():
                    f.write(str(item).strip())
                    f.write('\n')


def get_dict(dictionary_file_path):
    tokens = []
    with open(dictionary_file_path, 'r', newline='') as csvfile:
        lines = csv.reader(csvfile, delimiter='\n', quotechar='|')
        for row in lines:
            if len(row) == 0:
                continue
            tokens.append(row[0])
    return set(tokens)


def get_tokens_from_annotated_corpus(document_path):
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
