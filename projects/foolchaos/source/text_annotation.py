import logging
import os

from text_mining.annotation import annotate_text

logging.basicConfig(format="%(asctime)s -- %(message)s", level=logging.INFO)


def process(path, dir_name):
    counter = 1
    with open(path) as f:
        line = f.readline()
        while line:
            try:
                annotations = annotate_news_text(line=line)
                logging.info(f"text processing >>> {counter}")
                dir_path = f"../assets/{dir_name}/" + line.split('","')[0][1:] + "/"
                if not os.path.exists(dir_path):
                    logging.info(f"dir created {dir_path}")
                    os.makedirs(dir_path)
                f_out = open(dir_path + str(line.split('","')[1]) + '.tsv', 'w+')
                f_out.truncate(0)
                for annotation in annotations:
                    if annotation.token_type != "whitespace":
                        f_out.write(f"{annotation.token}\t{annotation.stem}\t{annotation.lemma}\n")
                f_out.close()
            except Exception as ex:
                logging.error(ex)
                logging.error(f"Error while processing file {path}")
            logging.info(f"done >>> {counter}\n")
            counter += 1
            line = f.readline()


def annotate_news_text(line):
    splitted = line.split('","')
    headers = annotate_text(splitted[1] + ".")
    news = annotate_text(splitted[2][:-2])
    return headers + news
