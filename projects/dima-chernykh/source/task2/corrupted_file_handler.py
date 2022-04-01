import os

from source.task2.file_helper import get_dict, get_tokens_from_annotated_corpus
from source.task2.typos_module import fix_token_typo


def print_score(valid_tokens_in_corrupted_file_cnt,
                correct_tokens_in_corrupted_file_before_cnt,
                correct_tokens_in_corrupted_file_after_cnt):
    print("Valid tokens in corrupted file: {}".format(valid_tokens_in_corrupted_file_cnt))
    print("Correct tokens in corrupted file before: {}".format(correct_tokens_in_corrupted_file_before_cnt))
    print("Correct tokens in corrupted file after: {}".format(correct_tokens_in_corrupted_file_after_cnt))
    print("Before typos module: {}"
          .format(correct_tokens_in_corrupted_file_before_cnt / valid_tokens_in_corrupted_file_cnt))
    print("After typos module: {}"
          .format(correct_tokens_in_corrupted_file_after_cnt / valid_tokens_in_corrupted_file_cnt))


def handle_corrupted_file(path_to_corrupted_tokens_dataset, path_to_correct_tokens_dataset, path_to_token_dict):
    valid_tokens_in_corrupted_file_cnt = 0
    correct_tokens_in_corrupted_file_before_cnt = 0
    correct_tokens_in_corrupted_file_after_cnt = 0

    token_dict = get_dict(path_to_token_dict)
    fixed_token_cnt = 1
    token_counter_limit = 200
    for root, dirs, files in os.walk(path_to_corrupted_tokens_dataset):
        if fixed_token_cnt == token_counter_limit:
            break
        for file in files:
            if fixed_token_cnt == token_counter_limit:
                break
            corrupted_token_list = get_tokens_from_annotated_corpus(root + '/' + file)
            corrupted_token_list_cnt = len(corrupted_token_list)
            original_token_list = get_tokens_from_annotated_corpus(
                root.replace(path_to_corrupted_tokens_dataset, path_to_correct_tokens_dataset) + '/' + file)
            original_token_list_cnt = len(original_token_list)
            if corrupted_token_list_cnt != original_token_list_cnt:
                print("Skip: {}".format(root + '/' + file))
            else:
                valid_tokens_in_corrupted_file_cnt += corrupted_token_list_cnt
                for index, token in enumerate(corrupted_token_list):
                    if fixed_token_cnt == token_counter_limit:
                        break
                    token = token.split()[0]
                    fixed_token = fix_token_typo(token, token_dict)
                    if fixed_token == token:
                        correct_tokens_in_corrupted_file_before_cnt += 1
                    else:
                        corrupted_token_list[index] = fixed_token
                    fixed_token_cnt += 1
                for index, token in enumerate(corrupted_token_list):
                    if corrupted_token_list[index] == original_token_list[index]:
                        correct_tokens_in_corrupted_file_after_cnt += 1
    print_score(
        valid_tokens_in_corrupted_file_cnt,
        correct_tokens_in_corrupted_file_before_cnt,
        correct_tokens_in_corrupted_file_after_cnt)
