import nltk

from text_annotation import process


def main():
    # set up:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('omw-1.4')

    # lab - 1
    process('../assets/train.csv', 'train')
    process('../assets/test.csv', 'test')


if __name__ == "__main__":
    main()
