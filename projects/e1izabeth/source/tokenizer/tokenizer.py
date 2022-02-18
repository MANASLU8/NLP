from xml.sax import saxutils
import re

knownAbbrevs = {
    "st.": "saint",
    "dr.": "doctor",
    "tel.": "telephone",
    "no.": "number",
    "u.s.": "United States",
    "prof.": "professor",
    "inc.": "incorporated",
    "ltd.": "limited",
    "corp.": "corporation",
    "mr.": "mister"
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
    ["url", "[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"],
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
                    break;
                else:
                    print('failed to tokenize: ' + s)
        else:
            print('failed to tokenize: ' + s)
    return line


