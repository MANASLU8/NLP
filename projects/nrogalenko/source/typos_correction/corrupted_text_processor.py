import os
import glob
from source.text_annotation.tokenizer import tokenize
from .tokens_dictionary_creator import get_words_dictionary_list
from .file_utils import get_tokens_list_from_document_annotation
from .typos_correction_module import fix_typo_with_dictionary


correct_tokens_in_corrupted_file_before_module = 0
correct_tokens_in_corrupted_file_after_module = 0


def process_corrupted_file(path, annotation_documents_dir_path):
    global correct_tokens_in_corrupted_file_before_module, correct_tokens_in_corrupted_file_after_module
    word_dictionary_list = get_words_dictionary_list(annotation_documents_dir_path, "../assets/word-dictionary/test/dict.json")
    correct_documents_tokens_sum_amount = 0
    lines_counter = 1
    word_dictionary_set = set(word_dictionary_list)
    with open(path) as f:
        line_raw = f.readline()
        while line_raw:
            try:
                line = line_raw[:1] + "\"" + line_raw[1:]
                document_annotation_file_path = str(glob.glob(annotation_documents_dir_path + "/**/" + str(lines_counter) + ".tsv", recursive=True)[0])
                tokens_from_document_annotation = get_tokens_list_from_document_annotation(document_annotation_file_path)
                tokens_count_in_document_annotation = len(tokens_from_document_annotation)
                tokens_list = tokenize(line)
                tokens_list = [token for token in tokens_list if token.token_tag != "whitespace"]
                label = line.split('","')[0]
                print(lines_counter)
                # print(tokens_from_document_annotation)
                # print([token.token for token in tokens_list])
                if tokens_count_in_document_annotation != len(tokens_list):
                    print("Skip document with path " + document_annotation_file_path)
                else:
                    correct_documents_tokens_sum_amount += tokens_count_in_document_annotation
                    for token in tokens_list:
                        fixed_token = fix_typo_with_dictionary(token.token, word_dictionary_set)
                        if fixed_token == token.token:
                            correct_tokens_in_corrupted_file_before_module += 1
                        else:
                            token.token = fixed_token
                    # print("fixed corrupted tokenized: " + str([token.token for token in tokens_list]))
                    for i in range(len(tokens_list)):
                        if tokens_list[i].token == tokens_from_document_annotation[i]:
                            correct_tokens_in_corrupted_file_after_module += 1
                    # print("failed to fix " + tokens_list[i].token)
            except Exception as ex:
                print(ex)
                print("Error while processing file " + path)
            lines_counter += 1
            line_raw = f.readline()
        print("correct documents tokens sum number " + str(correct_documents_tokens_sum_amount))
        print("correct documents tokens in corrupted before module " + str(correct_tokens_in_corrupted_file_before_module))
        print("correct documents tokens in corrupted after module " + str(correct_tokens_in_corrupted_file_after_module))
        print("Before module: " + str(correct_tokens_in_corrupted_file_before_module / correct_documents_tokens_sum_amount)
              + " After module: " + str(correct_tokens_in_corrupted_file_after_module / correct_documents_tokens_sum_amount))
