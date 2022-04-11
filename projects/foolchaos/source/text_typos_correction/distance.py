from numpy import zeros

from .keyboard import get_keyboard_distance, max_keyboard_distance

NOP_COST = 0
REMOVAL_COST = 1
INSERTION_COST = 1
SUBSTITUTION_COST = 2


def get_weighted_cost(operation_cost, symbol1, symbol2):
    try:
        return operation_cost + (get_keyboard_distance(symbol1, symbol2) / max_keyboard_distance)
    except ValueError:
        return SUBSTITUTION_COST


def count_edit_distance(X, Y, is_weighted, is_matrix=None):
    N = len(X) + 1
    M = len(Y) + 1
    matrix = zeros(shape=(N, M))
    for i in range(1, N):
        matrix[i][0] = i * REMOVAL_COST
    for j in range(1, M):
        matrix[0][j] = j * INSERTION_COST
    for i in range(1, N):
        for j in range(1, M):
            final_SUBSTITUTION_COST = get_weighted_cost(
                SUBSTITUTION_COST - 1,
                X[i - 1],
                Y[j - 1]
            ) if is_weighted else SUBSTITUTION_COST

            matrix[i, j] = min(
                matrix[i - 1][j] + REMOVAL_COST,
                matrix[i][j - 1] + INSERTION_COST,
                matrix[i - 1][j - 1] + (final_SUBSTITUTION_COST if X[i - 1] != Y[j - 1] else NOP_COST)
            )
    if is_matrix is not None:
        return matrix
    return matrix[N - 1, M - 1]


def count_edit_distance_faster(x, y):
    X = list(x)
    Y = list(y)

    N = len(X)
    M = len(Y)

    result = []

    if N <= 0:
        return

    if M == 1:
        if X[0] in Y:
            result.append(X[0])
            print(result)
        return

    mid = N // 2

    f = count_edit_distance(X[0:mid + 1], Y, True, True)
    f = f[len(f) - 1]

    s = count_edit_distance(list(reversed(X[mid:N])), list(reversed(Y)), True, True)
    s = s[len(s) - 1]

    max = s[0]

    it_max = -1

    for j in range(0, M):
        if f[j] + s[M - (j + 1)] > max:
            max = f[j] + s[M - (j + 1)]
            it_max = j
    if f[M - 1] > max:
        it_max = M - 1
    count_edit_distance_faster(X[0:mid + 1], Y[0:it_max])
    count_edit_distance_faster(X[mid:N], Y[it_max + 1:M])


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
