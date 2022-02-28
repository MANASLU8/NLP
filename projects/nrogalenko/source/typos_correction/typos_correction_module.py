import numpy as np

removal_cost = 1
insertion_cost = 1
substitution_cost = 2
nop_cost = 0

keyboard_array = [
    ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\''],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/'],
    [' ']
    ]

caps_keyboard_array = [
    ['~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', '|'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', '"'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '?'],
    [' ']
    ]


def is_symbol_in_keyboard_array(symbol, array):
    return True in [symbol in row for row in array]


def get_keyboard_array(symbol):
    if is_symbol_in_keyboard_array(symbol, keyboard_array):
        return keyboard_array
    elif is_symbol_in_keyboard_array(symbol, caps_keyboard_array):
        return caps_keyboard_array
    else:
        raise ValueError(symbol + " not found in keyboard")


def get_keyboard_symbol_location(symbol, array):
    for array_row in array:
        if symbol in array_row:
            row = array.index(array_row)
            column = array_row.index(symbol)
            return row, column
    raise ValueError(symbol + " not found in given keyboard layout")


def get_keyboard_distance(symbol1, symbol2):
    coord1 = get_keyboard_symbol_location(symbol1, get_keyboard_array(symbol1))
    coord2 = get_keyboard_symbol_location(symbol2, get_keyboard_array(symbol2))
    return np.sqrt(np.square(coord1[0] - coord2[0]) + np.square(coord1[1] - coord2[1]))


max_keyboard_distance = get_keyboard_distance(keyboard_array[0][0], keyboard_array[-1][-1])


def get_weighted_cost(operation_cost, symbol1, symbol2):
    try:
        return operation_cost + (get_keyboard_distance(symbol1, symbol2) / max_keyboard_distance)
    except ValueError:
        return substitution_cost


def count_edit_distance(X, Y, is_weighted):
    N = len(X) + 1
    M = len(Y) + 1
    distance_matrix = np.zeros(shape=(N, M))
    for i in range(1, N):
        distance_matrix[i][0] = i * removal_cost
    for j in range(1, M):
        distance_matrix[0][j] = j * insertion_cost
    for i in range(1, N):
        for j in range(1, M):
            final_substitution_cost = get_weighted_cost(substitution_cost - 1, X[i-1], Y[j-1]) if is_weighted else substitution_cost
            distance_matrix[i, j] = min(
                distance_matrix[i-1][j] + removal_cost,
                distance_matrix[i][j-1] + insertion_cost,
                distance_matrix[i-1][j-1] + (final_substitution_cost if X[i-1] != Y[j-1] else nop_cost)
            )
    # print(distance_matrix)
    return distance_matrix[N-1, M-1]


def fix_typo_with_dictionary(token, tokens_dictionary_list):
    if token in tokens_dictionary_list:
        return token
    fixed_token = ""
    min_distance = 9999
    for correct_token in tokens_dictionary_list:
        distance = count_edit_distance(token, correct_token, True)
        if distance < min_distance:
            min_distance = distance
            fixed_token = correct_token
    return fixed_token

