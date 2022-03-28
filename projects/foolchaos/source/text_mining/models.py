class Token:
    def __init__(self, token, token_type):
        self.token = token
        self.token_type = token_type

    def __str__(self):
        return f"{self.token}\t{self.token_type}"

    def __eq__(self, other):
        return (
                       self.token == other.token
               ) and (
                       self.token_type == other.token_type
               )


class Annotation(Token):
    def __init__(self, token, stem, lemma, token_type):
        super().__init__(token, token_type)
        self.stem = stem
        self.lemma = lemma

    def __str__(self):
        return f"{self.token}\t{self.token_type}\t{self.stem}\t{self.lemma}"

    def __eq__(self, other):
        return (
                       self.token == other.token
               ) and (
                       self.stem == other.stem
               ) and (
                       self.lemma == other.lemma
               )
