import unittest
from scipy.spatial import distance
from source.text_vectorization.vectorize import vectorize_custom_text
from source.util.rle_encoder import rle_encode, rle_decode
from source.text_vectorization.cosine import cosine_distance


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

    def test_cosine_distance(self):
        vector1 = [1, 0, 0.5, 0.2, 0.4, 2]
        vector2 = [1, 2, 0.1, 0.8, 0.6, 1.2]
        self.assertEqual(round(distance.cosine(vector1, vector2), 4), round(cosine_distance(vector1, vector2), 4))


if __name__ == '__main__':
    unittest.main()
