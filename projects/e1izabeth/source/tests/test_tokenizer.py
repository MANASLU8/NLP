import unittest

from tokenizer.tokenizer import tokenize_text


class TestTokenization(unittest.TestCase):
    def test_on_empty_string(self):
        self.assertEqual(len(tokenize_text("")), 0)

    def test_on_tokens_count(self):
        self.assertEqual(len(tokenize_text("Hello, World!")), 5)

    def test(self):
        self.assertEqual(tokenize_text("Let's have a good day!"), [
            [0,'word',"Let's"],
            [5,'whitespace',' '],
            [6,'word','have'],
            [10,'whitespace',' '],
            [11,'word','a'],
            [12, 'whitespace', ' '],
            [13,'word','good'],
            [17, 'whitespace', ' '],
            [18,'word','day'],
            [21,'punct','!']])


if __name__ == "__main__":
    unittest.main()