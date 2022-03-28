import csv
import os
from pathlib import Path


def read_from_file(paths):
    path = Path(str(Path(Path.cwd()))[:-len("source")], "assets", "resources", paths)
    # print(path)
    files = []
    with open(path, 'r', newline='') as csvfile:
        lines = csv.reader(csvfile, delimiter='\n', quotechar='|')
        for row in lines:
            files.append(row)
    return files


def write_tokens(paths, tok, tags):
    # print(paths)
    path = (paths.split(sep="."))[0]
    for t in tok:
        new_path = Path(str(Path(Path.cwd()))[:-len("source")], "assets",
                        "annotated-corpus", path, t[0], t[1] + ".tsv")
        path_to_main_folders = Path(str(Path(Path.cwd()))[:-len("source")],
                                    "assets", "annotated-corpus", path)
        path_to_class_folders = Path(str(Path(Path.cwd()))[:-len("source")],
                                     "assets", "annotated-corpus", path, t[0])
        # print(new_path)
        if not os.path.exists(path_to_main_folders):
            os.mkdir(path_to_main_folders)
        if not os.path.exists(path_to_class_folders):
            os.mkdir(path_to_class_folders)
        with open(new_path, 'w+', newline="\n") as tsvfile:
            writer = csv.writer(tsvfile, dialect='excel-tab')
            for i in range(len(t[2])):
                if t[2][i][0] == "\n":
                    writer.writerow([])
                else:
                    if tags:
                        writer.writerow([t[2][i][0], t[3][i], t[4][i], t[2][i][1]])
                    else:
                        writer.writerow([t[2][i][0], t[3][i], t[4][i]])


def create_dict(path, d):
    new_path = Path(str(Path(Path.cwd()))[:-len("source")], "assets", "resources", path)
    # print(new_path)
    with open(new_path, 'w+', newline="\n") as csvfile:
        writer = csv.writer(csvfile)
        for x in d:
            writer.writerow([x])
