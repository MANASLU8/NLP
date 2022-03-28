import numpy as np

keyboard_array_mapping = [
    ['`~', '1!', '2@', '3#', '4$', '5%', '6^', '7&', '8*', '9(', '0)', '-_', '=+'],
    ['qQ', 'wW', 'eE', 'rR', 'tT', 'yY', 'uU', 'iI', 'oO', 'pP', '[{', ']}', '\\|'],
    ['aA', 'sS', 'dD', 'fF', 'gG', 'hH', 'jJ', 'kK', 'lL', ';:', '\'"'],
    ['zZ', 'xX', 'cC', 'vV', 'bB', 'nN', 'mM', ',<', '.>', '/?'],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
]


def is_symbol_in_keyboard_array(symbol, array):
    return True in [symbol in row for row in array]


def get_keyboard_symbol_location(symbol, array):
    for array_row in array:
        if symbol in array_row:
            row = array.index(array_row)
            column = array_row.index(symbol)
            return row, column
    raise ValueError("{} not found".format(symbol))


def get_keyboard_distance(symbol1, symbol2):
    coord1 = get_keyboard_symbol_location(symbol1, keyboard_array_mapping)
    coord2 = get_keyboard_symbol_location(symbol2, keyboard_array_mapping)
    return np.sqrt(np.square(coord1[0] - coord2[0]) + np.square(coord1[1] - coord2[1]))


def get_max_keyboard_distance():
    get_keyboard_distance(keyboard_array_mapping[0][0], keyboard_array_mapping[-1][-1])
