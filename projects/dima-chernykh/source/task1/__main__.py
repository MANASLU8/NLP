import os

from source.task1.lemmer import add_lemmas
from source.task1.stemmer import add_stems
from source.task1.tokenizer import tokenize


def replace_path(path, frm, to):
    pre, match, post = path.rpartition(frm)
    return ''.join((to if match else pre, match, post))


def tokenize_words():
    path = 'assets/dataset'
    for root, dirs, files in os.walk(path):
        for file in files:
            src_file = os.path.join(root, file)
            dist_path_dir = "assets/annotated-corpus/" + '/'.join(src_file.split("/")[2:-1])
            dist_file = os.path.split(src_file)[1] + ".tsv"
            print(dist_path_dir + "/" + dist_file)
            with open(src_file, "r", encoding="utf-8", errors="ignore") as f:
                try:
                    tokens = add_lemmas(add_stems(tokenize(f.read())))
                    tsv_columns_order = ["token", "stem", "lemma", "tag"]
                    tsv = tokens[tsv_columns_order].to_csv(sep="\t", index=False, header=False, line_terminator="\n")
                    tsv = tsv.replace("FAKE_END\tFAKE_END\tFAKE_END\tSENT_END", "")
                    tsv = tsv.replace("SENT_END", "SENT_END" + "\n")
                except KeyError:
                    if not os.path.exists(dist_path_dir):
                        os.makedirs(dist_path_dir)
                    with open(dist_path_dir + "/" + dist_file, "w") as f:
                        f.write(tsv)
                    continue
            if not os.path.exists(dist_path_dir):
                os.makedirs(dist_path_dir)
            with open(dist_path_dir + "/" + dist_file, "w") as f:
                f.write(tsv)


tokenize_words()