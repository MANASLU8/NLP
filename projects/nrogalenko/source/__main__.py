import nltk
from text_annotation.text_processor import process_file


def main():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    process_file('../assets/train.csv', 'train')
    process_file('../assets/test.csv', 'test')


if __name__ == "__main__":
    main()
