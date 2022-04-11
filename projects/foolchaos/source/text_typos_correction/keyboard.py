from numpy import sqrt, square

english_keyboard = [
    ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\''],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/'],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
]

caps_english_keyboard = [
    ['~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', '|'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', '"'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '?'],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
]


def is_symbol_in_keyboard_array(symbol, array):
    return True in [symbol in row for row in array]


def get_keyboard_array(symbol):
    if is_symbol_in_keyboard_array(symbol, english_keyboard):
        return english_keyboard
    elif is_symbol_in_keyboard_array(symbol, caps_english_keyboard):
        return caps_english_keyboard
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
    return sqrt(square(coord1[0] - coord2[0]) + square(coord1[1] - coord2[1]))


max_keyboard_distance = get_keyboard_distance(english_keyboard[0][0], english_keyboard[-1][-1])
