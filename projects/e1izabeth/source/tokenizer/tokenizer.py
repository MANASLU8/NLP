from xml.sax import saxutils
import re


tokens = [
    ["number", "[0-9]+(.|,|-)[0-9]*"],
    ["whitespace", "\\s|\\n|\\\\|\\t"],
    ["punct", ",|\\.|\\?|\\!|(\\.\\.\\.)|\\(|\\)"],
    ["quoted", "(\\\")[^\\\"]*(\\\")"],
    ["word", "[A-Za-z][A-Za-z\\']*(-[A-Z\\']?[A-Za-z\\']+)*"],
    #["abbrev", "[A-Z]{2,}"],
    ["metrics", "[0-9]+(-)?[A-Z]?[a-z]+"],
    ["other", ".[^a-zA-Z0-9]*"],
]

regex = re.compile("^(" + "|".join(map(lambda t: "(?P<" + t[0] + ">" + t[1] + ")", tokens)) + ")")

classes = dict([
    (1, 'World'),
    (2, 'Sports'),
    (3, 'Business'),
    (4, 'Sci-Tech')
])

def tokenize_text(text):
    data = saxutils.unescape(text)
    pos = 0
    s = text
    line = []
    while len(s) > 0:
        match = regex.search(s)
        if match and match.endpos > match.pos:
            for gr in tokens:
                tt = list(filter(lambda kv: kv[1] is not None, match.groupdict().items()))
                if len(tt) == 1:
                    line.append([pos, tt[0][0], tt[0][1]])
                    pos += len(tt[0][1])
                    s = s[len(tt[0][1]):]
                    break;
                else:
                    print('failed to tokenize: ' + s)
        else:
            print('failed to tokenize: ' + s)
    return line
