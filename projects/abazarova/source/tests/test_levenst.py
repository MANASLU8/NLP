import unittest

from projects.abazarova.source.lab2.levenst import *


class TestLevenst(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(0.25, distance('2', '@'))
        self.assertEqual(0.25, distance('g', 'G'))
        self.assertEqual(0.5, distance('s', 'd'))
        self.assertEqual(0.5, distance('s', 'D'))
        self.assertEqual(0.75, distance('s', 'e'))
        self.assertEqual(1, distance('s', 't'))
        self.assertEqual(1, distance('w', 'l'))

    def test_levenst(self):
        self.assertEqual(0.5, lev_hirsch("worf", "word"))
        self.assertEqual(0.25, lev_hirsch("worD", "word"))
        self.assertEqual(0.75, lev_hirsch("word", "dord"))
        self.assertEqual(1, lev_hirsch("worfd", "word"))
        self.assertEqual(1, lev_hirsch("world", "word"))
        self.assertEqual(5.0, lev_hirsch("qwert", "cvbnm"))
        self.assertEqual(2, lev_hirsch("wod", "word"))


if __name__ == "__main__":
    unittest.main()
