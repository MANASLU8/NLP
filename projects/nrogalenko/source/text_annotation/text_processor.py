import os
from .tokenizer import tokenize


def process_file(path, output_dir_name):
    lines_counter = 1
    with open(path) as f:
        line = f.readline()
        while line:
            try:
                tokens_list = tokenize(line)
                label = line.split('","')[0][1:]
                output_dir_path = "../assets/" + output_dir_name + "/" + label + "/"
                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path)
                f_out = open(output_dir_path + str(lines_counter) + '.tsv', 'w+')
                f_out.truncate(0)
                print(lines_counter)
                for token in tokens_list:
                    if token.token_tag != "whitespace":
                        f_out.write(str(token) + '\n')
                    if token.token in ['.', '!', '?', '...']:
                        # end of sentence
                        f_out.write('\n')
                f_out.close()
            except Exception as ex:
                print(ex)
                print("Error while processing file " + path)
            lines_counter += 1
            line = f.readline()

