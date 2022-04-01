import nltk

from text_annotation import process as token_process
from text_class import process_classification
from text_correction import process
from text_topic import process_topic
from text_vector import process_vectorization


def main():
    # set up:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('omw-1.4')

    # lab - 1
    token_process('../assets/train.csv', 'train')
    token_process('../assets/test.csv', 'test')

    # lab - 2
    process("../assets/test-corrupted.csv", "../assets/test")

    # lab - 3
    process_vectorization()

    # lab - 4
    process_topic()

    # lab - 5
    process_classification()


if __name__ == "__main__":
    main()
