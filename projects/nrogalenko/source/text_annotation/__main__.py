import nltk
import text_processor


def main():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    text_processor.process_file('../../assets/little-test.csv', 'little-test')


if __name__ == "__main__":
    main()
