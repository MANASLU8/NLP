import re
import os
import nltk
import pandas as pd
from pathlib import Path
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

end_of_clause = ['.', '?', '!']


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


knownAbbrevs = {
    "st.": "saint",
    "dr.": "doctor",
    "tel.": "telephone",
    "no.": "number",
    "u.s.": "United States",
    "u.k.": "United Kingdom",
    "prof.": "professor",
    "inc.": "incorporated",
    "ltd.": "limited",
    "corp.": "corporation",
    "co.": "corporation",
    "mr.": "mister",
    "plc.": "Public Limited Company",
    "assn.": "association",
    "univ.": "university",
    "intl.": "international",
    "sys.": "system",
    "est.": "Eastern Standard Time",
    "ext.": "extention",
    "sq.": "square",
    "jr.": "junior",
    "sr.": "senior",
    "bros.": "brothers",
    "ed.d.": "Doctor of Education",
    "ph.d.": "Doctor of Phylosophy",
    "sci.": "Science",
    "etc.": "Et Cetera",
    "al.": "al",
    "seq.": "sequence",
    "orig.": "original",
    "incl.": "include",
    "eg.": "eg",
    "avg.": "average",
    "pl.": "place",
    "min.": "min",
    "max.": "max",
    "cit.": "citizen",
    "mrs.": "mrs",
    "mx.": "mx",
    "miss.": "miss",
    "atty.": "attorney",
    "col.": "college",
    "messrs.": "messieurs",
    "gov.": "government",
    "adm.": "admiral",
    "rev.": "revolution",
    "fr.": "french",
    "maj.": "major",
    "sgt.": "sergeant",
    "cpl.": "corporal",
    "pvt.": "private",
    "capt.": "captain",
    "ave.": "avenue",
    "pres.": "president",
    "brig.": "brigadier",
    "cmdr.": "commander",
    "asst.": "assistant",
    "assoc.": "associate",
    "insp.": "inspiration"
}

tokens = [
    ["abbrev", "|".join(map(lambda kv: "(?i:" + re.escape(kv[0]) + ")", knownAbbrevs.items()))],
    ["ipaddress", "[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+"],
    ["numeral", "[0-9]+((th)|(\\'s))"],
    ["metrics", "[0-9]+(-)?[A-Z]?[a-z]+"],
    ["currency", "\\$[0-9]+(\\,[0-9]+)*"],
    ["number", "[0-9]+(.|,|-)[0-9]*"],
    ["whitespace", "\\s|\\n|\\\\|\\t"],
    ["braces", "\\(|\\)"],
    ["quoted", "(\\\")[^\\\"]*(\\\")"],
    ["url", "[-a-zA-Z0-9@:%._\\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\\+.~#?&//=]*)"],
    ["punct", ",|\\.|\\?|\\!|(\\.\\.\\.)"],
    ["word", "[A-Za-z][A-Za-z\\']*(-[A-Z\\']?[A-Za-z\\']+)*"],
    ["other", ".[^a-zA-Z0-9]*"]
]

regex = re.compile("^(" + "|".join(map(lambda t: "(?P<" + t[0] + ">" + t[1] + ")", tokens)) + ")")

classes = dict([
    (1, 'World'),
    (2, 'Sports'),
    (3, 'Business'),
    (4, 'Sci-Tech')
])


def tokenize_text(text):
    pos = 0
    s = text
    line = []
    while len(s) > 0:
        match = regex.search(s)
        if match and match.endpos > match.pos:
            for gr in tokens:
                tt = list(filter(lambda kv: kv[1] is not None, match.groupdict().items()))
                if len(tt) == 1:
                    kind = tt[0][0]
                    part = tt[0][1]
                    if kind == 'abbrev':
                        kind = 'word'
                        part = knownAbbrevs[part.lower()]
                    line.append([pos, kind, part])
                    pos += len(tt[0][1])
                    s = s[len(tt[0][1]):]
                    break
                else:
                    print('failed to tokenize: ' + s)
        else:
            print('failed to tokenize: ' + s)
    return line


def process_file(fname):
    print('working on ', fname)
    df = pd.read_csv(fname, sep=',', header=None)
    data = df.values
    data_count = len(data)
    n = 0
    for row in data:
        class_id = row[0]
        try:
            dir_path = "../assets/" + Path(fname).name.split('.')[0] + "/" + classes[class_id] + '/'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            f = open(dir_path + str(n) + '.tsv', 'w+')
            f.truncate(0)
            for i in range(1, len(row)):
                text = row[i]
                tokens = tokenize_text(text)
                prev = [0, '', '']
                for w in tokens:
                    if w[1] != 'whitespace':
                        f.write(w[2] + '\t' + stemmer.stem(w[2]) + "\t" + lemmatizer.lemmatize(w[2], get_wordnet_pos(w[2])) + '\n')
                    elif prev[2] in end_of_clause:
                        f.write('\n')
                    prev = w
                f.write('\n')
            f.close()
        except Exception as e:
            print(e)
            print([n, text, tokens])
            pass
        n = n + 1
        if n % 1000 == 0:
            print(int(n * 100 / data_count), '%')
