import numpy as np


def lev_vag_fish(seq1: str, seq2: str, hirsch=False):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            # если буквы равны, то смотрим:
            # либо удаление, либо вставка, либо замена, смотря что меньше
            matrix[x, y] = min(
                matrix[x - 1, y] + 1,
                matrix[x - 1, y - 1] + distance(seq1[x - 1], seq2[y - 1]),
                matrix[x, y - 1] + 1
            )
    # print(matrix)
    if hirsch:
        return matrix[size_x - 1]
    return matrix[size_x - 1, size_y-1]


def distance(ch1: str, ch2: str):
    if ch1 == ch2:
        return 0
    # Если соседние, то 0.5, по диагонали - 0.75, дальше - 1, та же кнопка - 0.25

    qwerty = [['`~', '1!', '2@', '3#', '4$', '5%', '6^', '7&', '8*', '9(', '0)', '-_'],
              ['Qq', 'Ww', 'Ee', 'Rr', 'Tt', 'Yy', 'Uu', 'Ii', 'Oo', 'Pp', '{[', '}}'],
              ['Aa', 'Ss', 'Dd', 'Ff', 'Gg', 'Hh', 'Jj', 'Kk', 'Ll', ':;', '\"\'', '+='],
              ['Zz', 'Xx', 'Cc', 'Vv', 'Bb', 'Nn', 'Mm', '<,', '>.', '?/', '|\\', '']]
    for i in range(3):
        for j in range(11):
            if ch1 in qwerty[i][j]:
                # eq
                if ch2 in qwerty[i][j]:
                    return 0.25
                # straight
                # up
                if i > 0 and ch2 in qwerty[i-1][j]:
                    return 0.5
                # down
                if i < 3 and ch2 in qwerty[i+1][j]:
                    return 0.5
                # left
                if j > 0 and ch2 in qwerty[i][j-1]:
                    return 0.5
                # right
                if j < 11 and ch2 in qwerty[i][j+1]:
                    return 0.5
                # diagonally
                # up left
                if i > 0 and j > 0 and ch2 in qwerty[i-1][j-1]:
                    return 0.75
                # up right
                if i > 0 and j < 11 and ch2 in qwerty[i-1][j+1]:
                    return 0.75
                # down left
                if i < 3 and j > 0 and ch2 in qwerty[i+1][j-1]:
                    return 0.75
                # down right
                if i < 3 and j < 11 and ch2 in qwerty[i+1][j+1]:
                    return 0.75
    return 1


# Смысл этой функции состоит в том, чтобы разделить строку на две группы,
# рекурсивно решить соответствующую строку соответствия
def lev_hirsch(x: str, y: str, score=0):
    # print("x: ", x, " y: ", y, " score b4: ", score)
    if len(x) == 0:
        score += len(x)
    elif len(y) == 0:
        score += len(y)
    elif len(x) == 1 or len(y) == 1:
        return lev_vag_fish(x, y)
    else:
        xmid = len(x) // 2
        ymid = len(y) // 2
        x1 = x[:xmid]
        y1 = y[:ymid]
        x2 = x[xmid:][::-1]
        y2 = y[ymid:][::-1]

        score = lev_hirsch(x1, y1) + lev_hirsch(x2, y2)

    return score
