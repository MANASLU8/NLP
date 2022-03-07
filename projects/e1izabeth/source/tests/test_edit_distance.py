import unittest

from typos.hirschberg import find_match
from typos.tdict import TreeDict


class TestTokenization(unittest.TestCase):

    def test_tree_correct(self):
        d = TreeDict()
        d.add('cat')
        d.add('dog')
        d.add('mouse')
        d.add('horse')

        r = d.match("mouse")
        self.assertEqual(r.naked_value().lower(), 'mouse')

    def test_tree_fix(self):
        d = TreeDict()
        d.add('cat')
        d.add('dog')
        d.add('mouse')
        d.add('horse')

        r = d.match("mhoue")
        self.assertEqual(r.naked_value().lower(), 'mouse')

    def test_hirshberg_fix(self):
        d = set()
        d.add('cat')
        d.add('dog')
        d.add('mouse')
        d.add('horse')

        a, b, s = find_match(d, 'mouse', 1)[0]
        self.assertEqual(a.lower(), 'mouse')

    def test_hirshberg_fix(self):
        d = set()
        d.add('cat')
        d.add('dog')
        d.add('mouse')
        d.add('horse')

        a, b, s = find_match(d, 'mhoue', 1)[0]
        self.assertEqual(a.lower(), 'mouse')


if __name__ == "__main__":
    unittest.main()
