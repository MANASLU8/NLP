import argparse
import os.path
import re
import time
import io
import math
from multiprocessing import Pool

import pandas as pd

from tokenizer.tokenizer.tokenizer import tokenize_text

qwerty = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
allchars = "".join(qwerty)
sim_matrix = dict(
    [(chr(ord('a') + y), dict([(chr(ord('a') + x), 0) for x in range(0, ord('z') - ord('a') + 1)])) for y in
     range(0, ord('z') - ord('a') + 1)])


def find_and_compute(lst, pred, f):
    for i, v in enumerate(lst):
        if pred(i, v):
            return f(i, v)
    return None


def fill_sim_pair(a, b, v):
    sim_matrix[a][b] = v
    sim_matrix[b][a] = v


def fill_sim_matrix():
    for a in allchars:
        x1, y1 = find_and_compute(qwerty, (lambda i, l: a in l),
                                  (lambda i, l: find_and_compute(l, (lambda j, c: c == a), (lambda j, c: (i, j)))))
        for b in allchars:
            x2, y2 = find_and_compute(qwerty, (lambda i, l: b in l),
                                      (lambda i, l: find_and_compute(l, (lambda j, c: c == b), (lambda j, c: (i, j)))))
            x = x2 - x1
            y = y2 - y1
            sim_matrix[b][a] = math.sqrt(x * x + y * y)

    fill_sim_pair('a', 'o', 1)
    fill_sim_pair('e', 'i', 1)
    fill_sim_pair('c', 'k', 1)
    fill_sim_pair('w', 'v', 1)


class TreeDictNode:
    def __init__(self):
        self.childs = dict()
        self.end = False


class Ctx:
    _id = 0

    def __init__(self, prev, pos, c, node, penalty, op, hd):
        Ctx._id = Ctx._id + 1
        self.id = Ctx._id
        self.prev = prev
        self.pos = pos
        self.char = c
        self.node = node
        self.penalty = penalty
        self.op = op
        self.hd = hd

    def naked_value(self):
        s = ""
        c = self
        while c.char is not None:
            if c.char != '-':
                s = c.char + s
            c = c.prev
        return s

    def value(self):
        s = ""
        c = self
        while c.char is not None:
            s = (c.char if c.op == 'in' else c.char.upper()) + s
            c = c.prev
        return s

    def path(self):
        s = ""
        c = self
        while c.char is not None:
            p = c.op + ' ' + c.char + '@' + str(c.pos)
            if c != self:
                p = p + ', '
            s = p + s
            c = c.prev
        return s

    def __str__(self):
        return self.value() + ' ~ ' + str(self.penalty) + ': { ' + self.path() + ' }'


class MatchResult:
    def __init__(self, best_by_word, all_results, best_result):
        self.best_by_word = best_by_word
        self.top_results = all_results
        self.best_match = best_result


class TreeDict:
    def __init__(self):
        self.root = TreeDictNode()

    def add(self, str):
        node = self.root
        for c in str:
            next = node.childs.get(c)
            if next is None:
                next = TreeDictNode()
                node.childs[c] = next
            node = next
        node.end = True

    def save(self, fname):
        f = open(fname, 'w')

        def store(n, p, w):
            f.write('{\n')
            if n.end:
                f.write(p + '  "_": "' + w + '",\n')
            for k, v in n.childs.items():
                f.write(p + '  "' + k + '": ')
                store(v, p + '  ', w + k)
            f.write(p + '},\n')

        store(self.root, '', '')
        f.close()

    def load(self, fname):
        print()

    def __extraCharPenalty(self, key, s, k):
        sk = sim_matrix[k]
        if s.pos == 0:
            p = sk[key[0]]
        elif s.pos >= len(key) - 1:
            p = sk[key[len(key) - 1]]
        else:
            p = min(sk[key[s.pos]], sk[key[s.pos + 1]])
        return p

    def __skipCharPenalty(self, key, s):
        sk = sim_matrix[key[s.pos]]
        if s.pos == 0:
            p = sk[key[1]]
        elif s.pos >= len(key) - 1:
            p = sk[key[len(key) - 1]]
        else:
            p = min(sk[key[s.pos - 1]], sk[key[s.pos + 1]])
        return p

    def match(self, key, best_by_word=False, top_matches_limit=0):
        active = [Ctx(None, 0, None, self.root, 0, '', 0)]
        all_results = []
        best_results = dict()
        best_result = None
        len_limit = len(key) * 2
        cnt = 0
        while len(active) > 0:
            cnt = cnt + 1
            s = active.pop()
            if s.pos <= len_limit and s.hd < 3 and s.penalty < 5:  # -- lookup depth limit

                # alternative char at pos
                if s.pos < len(key):
                    for (k, cn) in s.node.childs.items():
                        p = sim_matrix[k][key[s.pos]]
                        active.append(Ctx(s, s.pos + 1, k, cn, s.penalty + p, 'in' if p == 0 else 'replace',
                                          0 if p == 0 else s.hd + 1))
                # consider missing char k
                for (k, cn) in s.node.childs.items():
                    p = 2  # self.__extraCharPenalty(key, s, k)
                    active.append(Ctx(s, s.pos, k, cn, s.penalty + p, 'insert', s.hd + 1))
                # skip extra char at pos
                if s.pos < len(key):
                    p = 2  # self.__skipCharPenalty(key, s)
                    active.append(Ctx(s, s.pos + 1, '-', s.node, s.penalty + p, 'remove', s.hd + 1))

            if s.pos >= len(key) - 3 and s.node.end:
                if s.pos < len(key):
                    d = len(key) - s.pos
                    s = Ctx(s, s.pos, '-' * d, s.node, s.penalty + 2 * d, 'back*' + str(d), s.hd)
                if best_by_word:
                    value = s.naked_value()
                    old = best_results.get(value)
                    if old is None or old.penalty > s.penalty:
                        best_results[value] = s
                if top_matches_limit > 0:
                    all_results.append(s)
                    if len(all_results) > top_matches_limit:
                        all_results.sort(key=lambda c: c.penalty)
                        all_results.pop()
                if best_result is None or s.penalty < best_result.penalty:
                    best_result = s

        # print(len(all_results), ' of ',  cnt, ':')
        # all_results.sort(key=lambda c: c.penalty)
        # for r in all_results:
        #    print(r)
        # print()
        ret = list(best_results.values())
        ret.sort(key=lambda c: c.penalty)
        return MatchResult(ret, all_results, best_result)


def test():
    d = TreeDict()
    d.add('cat')
    d.add('dog')
    d.add('mouse')
    d.add('horse')
    # d.save('d:\\temp\\test.txt')

    # r = d.match("mhoue")
    r = d.match("mouse")
    for c in r:
        print(c)


def load_dictionary():
    d = TreeDict()
    df = 'D:\\github.com\\NLP\\projects\\e1izabeth\\assets\\dictionary-dedup'
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
                d.add(line.lower())
            line = f.readline()
    return d


def do_interactive():
    s = input('> ')
    while len(s) > 0 and s != 'q':
        m = d.match(s.strip().lower(), best_by_word=True).best_by_word
        m.reverse()
        for c in m:
            print(c)
        s = input('> ')


def handle_token(w):
    if w[1] == 'word' and re.match('^[a-zA-Z]+$', w[2]):
        mr = d.match(w[2].lower()).best_match
        if mr is not None:
            fw = mr.naked_value()
            if w[2][0].isupper():
                fw = fw[0].upper() + fw[1:]
            w[2] = fw
    return w


def process_file(d, fname):
    print('working on ', fname)
    df = pd.read_csv(fname, sep=',', header=None)
    with Pool(8) as p:
        with open(fname + '.td.out', 'w+') as f:
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
                        # w = handle_token(w)
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
    d = load_dictionary()
    do_interactive()
    #process_file(d, 'D:\\github.com\\NLP\\projects\\e1izabeth\\assets\\raw-dataset\\test-corrupted.csv')

