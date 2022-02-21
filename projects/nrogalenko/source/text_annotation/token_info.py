class TokenInfo:

    def __init__(self, token, stem, lemma, pos_tag, token_tag):
        self.token = token
        self.stem = stem
        self.lemma = lemma
        self.pos_tag = pos_tag
        self.token_tag = token_tag

    def __str__(self):
        return f"{self.token}    {self.stem}    {self.lemma}    {self.pos_tag}    {self.token_tag}"

    def __eq__(self, other):
        return self.token == other.token and self.stem == other.stem and self.lemma == other.lemma\
               and self.pos_tag == other.pos_tag and self.token_tag == other.token_tag
