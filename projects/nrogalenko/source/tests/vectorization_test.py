import unittest
from source.text_vectorization.vectorize import vectorize_custom_text
from source.util.rle_encoder import rle_encode, rle_decode


class VectorizationTest(unittest.TestCase):
    def test_tf_idf_vector(self):
        tokens_dict = {"apple": 2, "dog": 1, "cat": 1, "ball": 2, "door": 1, "window": 1}
        self.assertEqual(
            [0.0, 0.075, 0.0, 0.0, 0.075, 0.075],
            vectorize_custom_text("dog, ball door window", tokens_dict, "../../assets/t.csv")
        )

    def test_rle_encode(self):
        self.assertEqual(",2_0,1,999,2_888,10,1,0,3_22,2_1,3", rle_encode(",0,0,1,999,888,888,10,1,0,22,22,22,1,1,3"))

    def test_rle_decode(self):
        self.assertEqual([0, 0, 0, 11, 11, 10, 1, 0, 1, 1], rle_decode(",3_0,2_11,10,1,0,2_1"))


if __name__ == '__main__':
    unittest.main()
