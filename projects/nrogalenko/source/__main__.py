import nltk
from text_annotation.text_processor import process_file
from typos_correction.corrupted_text_processor import process_corrupted_file


def main():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    # process_file('../assets/train.csv', 'train')
    # process_file('../assets/test.csv', 'test')
    process_corrupted_file("../assets/test-corrupted.csv", "../assets/test")
    # fix_typo_with_dictionary("set", ["sat", "srt", "sxt"])


if __name__ == "__main__":
    main()
