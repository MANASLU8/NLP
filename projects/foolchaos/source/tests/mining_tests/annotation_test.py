from unittest import TestCase

from nltk import download

from projects.foolchaos.source.text_mining.annotation import annotate_text
from projects.foolchaos.source.text_mining.models import Annotation


class AnnotationTest(TestCase):
    def setUp(self):
        download('wordnet')
        download('averaged_perceptron_tagger')
        download('punkt')
        download('stopwords')
        download('omw-1.4')

    def test_annotation(self):
        annotations = annotate_text(
            text="Let's start to eat these 1001 burgers, but after need to run 60kms to example.com and type 'hello' in word."
        )
        expected_result = [
            Annotation("Let's", "Let", "Let's", "word"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("start", "start", "start", "word"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("to", "to", "to", "word"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("eat", "eat", "eat", "word"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("these", "these", "these", "word"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("1001", "1001", "1001", "number"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("burgers", "burger", "burger", "word"),
            Annotation(",", ",", ",", "punctuation sign"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("but", "but", "but", "word"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("after", "after", "after", "word"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("need", "need", "need", "word"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("to", "to", "to", "word"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("run", "run", "run", "word"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("60kms", "60kms", "60kms", "metrics"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("to", "to", "to", "word"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("example.com", "example.com", "example.com", "website"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("and", "and", "and", "word"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("type", "type", "type", "word"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("'hello'", "hello", "'hello'", "quotation"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("in", "in", "in", "word"),
            Annotation(" ", " ", " ", "whitespace"),
            Annotation("word", "word", "word", "word"),
            Annotation(".", ".", ".", "punctuation sign"),
        ]
        self.assertEqual(annotations, expected_result)
