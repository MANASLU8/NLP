import glob
import logging

from text_mining.annotation import annotate_text
from text_typos_correction.dictionary import get_words_dictionary_list
from text_typos_correction.distance import fix_typo_with_dictionary

TOKENS_BEFORE = 0
TOKENS_AFTER = 0

logging.basicConfig(format="%(asctime)s -- %(message)s", level=logging.INFO)


def tokenize_news_text(line):
    splitted = line.split('","')
    headers = annotate_text(f"{splitted[1]}.")
    news = annotate_text(f"{splitted[2][:-2]}")
    return headers + news


def get_tokens_from_document_annotation(document_path):
    tokens = []
    with open(document_path, "r") as f:
        line = f.readline()
        while line:
            if line:
                token = line.split("\t")[0]
                if token != '\n':
                    tokens.append(token)
            line = f.readline()
    f.close()
    return tokens


def process(path, annotation_documents_dir_path):
    global TOKENS_BEFORE, TOKENS_AFTER

    word_dictionary_list = get_words_dictionary_list(
        annotation_documents_dir_path,
        "../assets/word-dictionary/lol/dict.json"
    )

    amnt = 0
    counter = 1
    word_dictionary_set = set(word_dictionary_list)

    with open(path) as f:
        line_raw = f.readline()
        while line_raw:
            try:
                raw_1 = line_raw[:1]
                raw_2 = line_raw[1:]
                line = raw_1 + "\"" + raw_2
                document_annotation_file_path = str(
                    glob.glob(
                        annotation_documents_dir_path + "/**/" + str(counter) + ".tsv",
                        recursive=True
                    )[0]
                )
                tokens_from_document_annotation = get_tokens_from_document_annotation(
                    document_annotation_file_path
                )
                annotation = len(tokens_from_document_annotation)
                tokens = tokenize_news_text(line)
                tokens = [token for token in tokens if token.token_type != "whitespace"]

                logging.info(annotation == len(tokens))
                logging.info(tokens_from_document_annotation)
                logging.info([token.token for token in tokens])

                if annotation != len(tokens):
                    logging.info("SKIPPED >>> " + document_annotation_file_path)
                    pass
                else:
                    amnt += int(annotation)
                    for token in tokens:
                        if fixed_token == token.token:
                            TOKENS_BEFORE += 1
                        else:
                            token.token = fixed_token
                        fixed_token = fix_typo_with_dictionary(token.token, word_dictionary_set)
                    for i in range(len(tokens)):
                        if tokens[i].token == tokens_from_document_annotation[i]:
                            TOKENS_AFTER += 1
            except Exception as ex:
                logging.info(ex)
                logging.info("Error while processing file " + path)
            counter += 1
            line_raw = f.readline()
        logging.info(
            "sum number ",
            str(amnt)
        )
        logging.info(
            "before module ",
            str(TOKENS_BEFORE)
        )
        logging.info(
            "after module ",
            str(TOKENS_AFTER)
        )
        logging.info(
            "Before module TOTAL: ",
            str(TOKENS_BEFORE / amnt),
            " After module TOTAL: ",
            str(TOKENS_AFTER / amnt)
        )
