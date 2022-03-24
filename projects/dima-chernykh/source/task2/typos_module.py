import numpy as np

from source.task2.keyboard import get_keyboard_distance, get_max_keyboard_distance

removal_cost = 1
insertion_cost = 1
substitution_cost = 2
no_op_cost = 0


def get_weighted_cost(replace_cost, symbol1, symbol2):
    try:
        return replace_cost + (get_keyboard_distance(symbol1, symbol2) / get_max_keyboard_distance())
    except ValueError:
        return substitution_cost


def count_edit_distance(word1, word2):
    n = len(word1) + 1
    m = len(word2) + 1
    matrix = np.zeros((n, m))
    for i in range(1, n):
        matrix[i][0] = i * removal_cost
    for j in range(1, m):
        matrix[0][j] = j * insertion_cost
    for i in range(1, n):
        for j in range(1, m):
            matrix[i, j] = min(
                matrix[i - 1][j] + removal_cost,
                matrix[i][j - 1] + insertion_cost,
                matrix[i - 1][j - 1] + (get_weighted_cost(
                    substitution_cost - 1,
                    word1[i - 1],
                    word2[j - 1])
                )
                if word1[i - 1] != word2[j - 1] else no_op_cost)
    return matrix[n - 1, m - 1]


def fix_token_typo(token, tokens_dictionary_list):
    if token in tokens_dictionary_list:
        return token
    min_distance = 10000
    fixed_token = ""
    for correct_token in tokens_dictionary_list:
        distance = count_edit_distance(token, correct_token)
        if distance < min_distance:
            min_distance = distance
            fixed_token = correct_token
    return fixed_token
