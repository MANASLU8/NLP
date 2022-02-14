import csv
from pathlib import Path


def read_from_file(paths):
    path = Path(str(Path(Path.cwd()))[:-len("source.lab1")], "assets", "resources", paths)
    # print(path)
    file = []
    with open(path, 'r', newline='') as csvfile:
        lines = csv.reader(csvfile, delimiter='\n', quotechar='|')
        for row in lines:
            file.append(row)
    return file


def write_to_file(paths, d, tags):
    # print(paths)
    path = (paths.split(sep="."))[0]
    new_path = Path(str(Path(Path.cwd()))[:-len("source.lab1")], "assets", "annotated-corpus", path+".tsv")
    # print(new_path)
    # print(csv.list_dialects())
    with open(new_path, 'w', newline="\n") as tsvfile:
        writer = csv.writer(tsvfile, dialect='excel-tab')
        for t in sorted(d):
            if(tags):
                writer.writerow([t, d[t][1], d[t][2], d[t][0]])
            else:
                writer.writerow([t, d[t][1], d[t][2]])
