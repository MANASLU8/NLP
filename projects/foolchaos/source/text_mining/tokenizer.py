import re

from .models import Token


def tokenize(text, patterns):
    tokens = []
    while len(text) > 0:
        for pattern in patterns:
            matcher = re.search(r'^(' + patterns[pattern] + ')', text)
            if matcher:
                token = matcher.group()
                tokens.append(
                    Token(token=token, token_type=pattern)
                )
                text = text[len(token):]
                break
    return tokens
