import unittest
from source.text_annotation.tokenizer import tokenize
from source.text_annotation.token_info import TokenInfo


class TestTokenization(unittest.TestCase):
    def test_tokenization(self):
        self.assertEqual(tokenize("Let's check this 1 sentence, 60km test.com 'a' b."),
                         [TokenInfo("Let's", "Let", "Let's", "n", "word"),
                          TokenInfo(" ", " ", " ", "n", "whitespace"),
                          TokenInfo("check", "check", "check", "v", "word"),
                          TokenInfo(" ", " ", " ", "n", "whitespace"),
                          TokenInfo("this", "this", "this", "n", "word"),
                          TokenInfo(" ", " ", " ", "n", "whitespace"),
                          TokenInfo("1", "1", "1", "n", "number"),
                          TokenInfo(" ", " ", " ", "n", "whitespace"),
                          TokenInfo("sentence", "sentenc", "sentence", "n", "word"),
                          TokenInfo(",", ",", ",", "n", "punctuation sign"),
                          TokenInfo(" ", " ", " ", "n", "whitespace"),
                          TokenInfo("60km", "60km", "60km", "n", "metrics"),
                          TokenInfo(" ", " ", " ", "n", "whitespace"),
                          TokenInfo("test.com", "test.com", "test.com", "n", "website"),
                          TokenInfo(" ", " ", " ", "n", "whitespace"),
                          TokenInfo("'a'", "a", "'a'", "n", "quotation"),
                          TokenInfo(" ", " ", " ", "n", "whitespace"),
                          TokenInfo("b", "b", "b", "n", "word"),
                          TokenInfo(".", ".", ".", "n", "punctuation sign"),
                          ])


if __name__ == '__main__':
    unittest.main()
