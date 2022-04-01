import unittest

from typos.hirschberg import find_match
from typos.tdict import TreeDict


class TestTokenization(unittest.TestCase):

    def test_tree_correct(self):
        d = TreeDict()
        d.add('cat')
        d.add('dog')
        d.add('horse')
        d.add('mouse')

        r = d.match("mouse")
        self.assertEqual(r.best_match.naked_value(), 'mouse')

    def test_tree_fix(self):
        d = TreeDict()
        d.add('cat')
        d.add('dog')
        d.add('mouse')
        d.add('horse')

        r = d.match("mhoue")
        self.assertEqual(r.best_match.naked_value(), 'horse')

    def test_hirshberg_correct(self):
        d = set()
        d.add('cat'.upper())
        d.add('dog'.upper())
        d.add('mouse'.upper())
        d.add('horse'.upper())

        a, b, s = find_match(d, 'mouse'.upper(), 0)
        self.assertEqual(a.upper(), 'mouse'.upper())

    def test_hirshberg_fix(self):
        d = set()
        d.add('cat'.upper())
        d.add('dog'.upper())
        d.add('mouse'.upper())
        d.add('horse'.upper())

        a, b, s = find_match(d, 'mhoue'.upper(), 0)
        self.assertEqual(a.upper(), 'horse'.upper())


if __name__ == "__main__":
    unittest.main()
