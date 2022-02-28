import unittest
from source.typos_correction.typos_correction_module import count_edit_distance


class TestEditDistanceCount(unittest.TestCase):
    def test_distance_no_weight(self):
        self.assertEqual(count_edit_distance("execution", "intention", False), 8.0)


if __name__ == '__main__':
    unittest.main()
