import json
import os


def create_dictionary_file(token_files_dir_path, resulting_dictionary_path):
    tokens_dictionary_list = []
    for root, dirs, files in os.walk(token_files_dir_path):
        for file in files:
            with open(os.path.join(root, file), "r") as f:
                line = f.readline()
                while line:
                    if line:
                        tokens_dictionary_list.append(line.split("\t")[0])
                    line = f.readline()
            f.close()
    os.makedirs(os.path.dirname("/".join((resulting_dictionary_path + "/").split("/")[:-1])), exist_ok=True)
    with open(resulting_dictionary_path, "w") as f:
        f.write(json.dumps(tokens_dictionary_list))
        f.close()
    return tokens_dictionary_list


def get_words_dictionary_list(token_files_dir_path, dictionary_file_path):
    if os.path.exists(dictionary_file_path):
        with open(dictionary_file_path, 'r') as f:
            tokens_dictionary_list = json.loads(f.read())
    else:
        tokens_dictionary_list = create_dictionary_file(token_files_dir_path, dictionary_file_path)
    return tokens_dictionary_list
