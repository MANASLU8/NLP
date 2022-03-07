import unittest

from projects.abazarova.source.lab1.reader import *


class TestReader(unittest.TestCase):
    def setUp(self):
        self.paths = Path(str(Path(Path.cwd()))[:-len("source")], "assets", "resources", "try.csv")

    def test_read_from_file(self):
        self.assertEqual(len(read_from_file(self.paths)), 2)


if __name__ == "__main__":
    unittest.main()
