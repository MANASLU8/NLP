import unittest
from source.typos_correction.typos_correction_module import count_edit_distance, fix_typo_with_dictionary, get_keyboard_distance


class TestEditDistanceCount(unittest.TestCase):
    def test_distance_no_weight(self):
        self.assertEqual(count_edit_distance("execution", "intention", False), 8.0)

    def test_distance_weight(self):
        self.assertEqual(count_edit_distance("set", "srt", True), 1.25)

    def test_typo_fix_with_module(self):
        dictionary = ("applt", "apple", "test")
        self.assertEqual(fix_typo_with_dictionary("applw", dictionary), "apple")

    def test_keyboard_distance(self):
        self.assertEqual(get_keyboard_distance("q", "w"), 1)


if __name__ == '__main__':
    unittest.main()
