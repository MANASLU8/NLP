import os.path
import re
from multiprocessing import Pool

import pandas as pd

from mytokenizer.tokenizer import tokenize_text
from typos.__main__ import DICT_NAME

gap_penalty = dict([(chr(ord('A') + x), 2) for x in range(0, ord('Z') - ord('A') + 1)])
qwerty = ["qwertyuiop".upper(), "asdfghjkl".upper(), "zxcvbnm".upper()]
sim_matrix = dict(
    [(chr(ord('A') + y), dict([(chr(ord('A') + x), 0) for x in range(0, ord('Z') - ord('A') + 1)])) for y in
     range(0, ord('Z') - ord('A') + 1)])


def find_and_compute(lst, pred, f):
    for i, v in enumerate(lst):
        if pred(i, v):
            return f(i, v)
    return None


def fill_sim_pair(a, b, v):
    a = a.upper()
    b = b.upper()
    sim_matrix[a][b] = v
    sim_matrix[b][a] = v


def fill_sim_matrix():
    allchars = "".join(qwerty)
    for a in allchars:
        x1, y1 = find_and_compute(qwerty, (lambda i, l: a in l),
                                  (lambda i, l: find_and_compute(l, (lambda j, c: c == a), (lambda j, c: (i, j)))))
        for b in allchars:
            x2, y2 = find_and_compute(qwerty, (lambda i, l: b in l),
                                      (lambda i, l: find_and_compute(l, (lambda j, c: c == b), (lambda j, c: (i, j)))))
            x = x2 - x1
            y = y2 - y1
            sim_matrix[b.upper()][a.upper()] = x * x + y * y

    fill_sim_pair('a', 'o', 1)
    fill_sim_pair('e', 'i', 1)
    fill_sim_pair('c', 'k', 1)
    fill_sim_pair('w', 'v', 1)

def needlman_wunsch(A, B):
    # fill zero-matrix
    n, m = len(A), len(B)
    mat = []
    for i in range(n + 1):
        mat.append([0] * (m + 1))
    for j in range(1, m + 1):
        mat[0][j] = gap_penalty[B[j - 1]]
        mat[0][j] += mat[0][j - 1]
    for i in range(1, n + 1):
        mat[i][0] = gap_penalty[A[i - 1]]
        mat[i][0] += mat[i - 1][0]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            mat[i][j] = min(mat[i - 1][j - 1] + sim_matrix[A[i - 1]][B[j - 1]], mat[i][j - 1] + gap_penalty[B[j - 1]],
                            mat[i - 1][j] + gap_penalty[A[i - 1]])

    # find final alignment by backtracking through matrix
    alignment_a = ""
    alignment_b = ""
    i, j = n, m
    while i and j:
        score, score_diag, score_up, score_left = mat[i][j], mat[i - 1][j - 1], mat[i - 1][j], mat[i][j - 1]
        if score == score_diag + sim_matrix[A[i - 1]][B[j - 1]]:
            alignment_a = A[i - 1] + alignment_a
            alignment_b = B[j - 1] + alignment_b
            i -= 1
            j -= 1
        elif score == score_up + gap_penalty[A[i - 1]]:
            alignment_a = A[i - 1] + alignment_a
            alignment_b = '-' + alignment_b
            i -= 1
        elif score == score_left + gap_penalty[B[j - 1]]:
            alignment_a = '-' + alignment_a
            alignment_b = B[j - 1] + alignment_b
            j -= 1
    while i:
        alignment_a = A[i - 1] + alignment_a
        alignment_b = '-' + alignment_b
        i -= 1
    while j:
        alignment_a = '-' + alignment_a
        alignment_b = B[j - 1] + alignment_b
        j -= 1
    return [alignment_a, alignment_b, mat[n][m]]


def prefix_subroutine(x, y):
    n, m = len(x), len(y)
    mat = []
    for i in range(n + 1):
        mat.append([0] * (m + 1))
    for j in range(1, m + 1):
        mat[0][j] = gap_penalty[y[j - 1]]
        mat[0][j] += mat[0][j - 1]
    for i in range(1, n + 1):
        mat[i][0] = mat[i - 1][0] + gap_penalty[x[i - 1]]
        for j in range(1, m + 1):
            mat[i][j] = min(mat[i - 1][j - 1] + sim_matrix[x[i - 1]][y[j - 1]],
                            mat[i - 1][j] + gap_penalty[x[i - 1]],
                            mat[i][j - 1] + gap_penalty[y[j - 1]])

        mat[i - 1] = []
    return mat[n]


def suffix_subroutine(x, y):
    n, m = len(x), len(y)
    mat = []
    for i in range(n + 1):
        mat.append([0] * (m + 1))
    for j in range(m + 1):
        mat[0][j] = gap_penalty[y[j - 1]]
        mat[0][j] += mat[0][j - 1]
    for i in range(1, n + 1):
        mat[i][0] = mat[i - 1][0] + gap_penalty[x[i - 1]]
        for j in range(1, m + 1):
            mat[i][j] = min(mat[i - 1][j - 1] + sim_matrix[x[n - i]][y[m - j]],
                            mat[i - 1][j] + gap_penalty[x[i - 1]],
                            mat[i][j - 1] + gap_penalty[y[j - 1]])

        mat[i - 1] = []
    return mat[n]


def hirschberg(x, y):
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return needlman_wunsch(x, y)
    else:
        # make partitions
        F, B = prefix_subroutine(x[:n // 2], y), suffix_subroutine(x[n // 2:], y)
        partition = [F[j] + B[m - j] for j in range(m + 1)]
        cut = partition.index(min(partition))

        F, B, partition = [], [], []
        call_left = hirschberg(x[:n // 2], y[:cut])
        call_right = hirschberg(x[n // 2:], y[cut:])
        # [alignment 1, alignment 2, similarity]
        return [call_left[r] + call_right[r] for r in range(3)]


def chars2indexes(str):
    return [(ord(str[i]) - ord('a')) for i in range(0, len(str))]


def check():
    dd = ['cat', 'dog', 'mouse', 'horse']
    for d in dd:
        a, b, s = hirschberg(d.upper(), "mhoue".upper())
        print(a)
        print(len(a) * '|')
        print(b)
        print(' ~ ', s)
        print()


def test():
    check()
    str1 = 'cat'
    str2 = 'cat'

    fill_sim_matrix()
    AA, BB, ss = hirschberg(str1.upper(), str2.upper())

    print("score: ", ss)
    print(AA)
    print(len(AA) * '|')
    print(BB)
    print()


def load_dict(name):
    ww = set()
    df = '../assets/' + name
    sz = os.path.getsize(df)
    with open(df) as f:
        print('loading ', df)
        line = f.readline()
        n = 0
        while line:
            n = n + 1
            if n % 10000 == 0:
                print('\t', f.tell() / sz * 100, '%')
            line = line.strip()
            if re.match('^[a-zA-Z]+$', line):
                ww.add(line.upper())
            line = f.readline()
    dd = dict()
    for word in ww:
        l = len(word)
        wd = dd.get(l)
        if wd is None:
            wd = []
            dd[l] = wd
        wd.append(word)

    return dd


def write_dict(ww, fname):
    with open(fname, 'w+') as f:
        f.truncate(0)
        for w in ww:
            f.write(w)
            f.write('\n')


def find_match(dictionary, key, count=0):
    results = []
    best_result = None
    if key in dictionary:
        return key, key, 0
    else:
        for word in dictionary:
            #n = n + 1
            #if n%100 == 0:
                #print('\t', n / len(dictionary) * 100, '%')
            p = hirschberg(word, key)
            if best_result is None or p[2] < best_result[2]:
                best_result = p
            if count > 0:
                results.append(p)
                if len(results) > count:
                    results.sort(key=lambda c: c[2])
                    results.pop()
    if count > 0:
        results.sort(key=lambda c: c[2])
        results.reverse()
        return results
    else:
        return best_result


def do_interactive(dictionary):
    s = input('> ')
    while len(s) > 0 and s != 'q':
        s = s.strip().upper()
        results = find_match(dictionary, s, 20)
        for (AA, BB, ss) in results:
            print('\t', AA, ' ~ ', ss)
        s = input('> ')


def handle_token(w):
    if w[1] == 'word' and re.match('^[a-zA-Z]+$', w[2]):
        sw = w[2].upper()
        a, b, s = find_match(dictionary, sw)
        if sw != a:
            fw = a.lower()
            if w[2][0].isupper():
                fw = fw[0].upper() + fw[1:]
            w[2] = fw.replace('-', '')
    return w


def process_file(d, fname):
    print('working on ', fname)
    df = pd.read_csv(fname, sep=',', header=None)
    with Pool(8) as p:
        with open(fname + '.h.out', 'w+') as f:
            f.truncate(0)
            data = df.values
            data_count = len(data)
            n = 0
            for row in data:
                class_id = row[0]
                f.write(str(class_id))
                try:
                    sc = 0
                    for i in range(1, len(row)):
                        sc = sc + 1
                        text = row[i]
                        tokens = tokenize_text(text)
                        # wc = 0
                        tokens = p.map(handle_token, tokens)
                        # wc = wc + 1
                        # print('\t[', str(sc), '/', str(len(row)), ']', str(wc), '/', str(len(tokens)))
                        f.write(',"')
                        f.write(''.join(list(map(lambda t: t[2].replace('"', '""'), tokens))))
                        f.write('"')
                        f.flush()
                except Exception as e:
                    print(e)
                    print([n, text, tokens])
                    pass

                f.write('\n')
                n = n + 1
                print(str(n * 100 / data_count), '% (', n, '/', data_count, ')')


if __name__ == '__main__':
    fill_sim_matrix()
    dict_name = '../assets/dictionary-dedup'
    # writeDict(loadDict(), '../assets/dictionary-dedup')
    dictionary = load_dict('../assets/' + DICT_NAME)
    process_file(dictionary, '../assets/raw-dataset/test-corrupted.csv')
    # do_interactive()
