from .lemmatizer import lemmatize
from .models import Annotation
from .stemmer import stem
from .tokenizer import tokenize

patterns = {
    "abbreviations": r'(?i)(((calif|ky|inc|corp|st|dr|tel|no|ltd|co|mr|mrs|etc|all)\.)|(u\.s\.|u\.k\.|e\.g\.?)|(('
                     r'?<=\s)[a-z]{1,5}\.(?=\s[a-z])))',
    "phone": r'(\([0-9]{3}\)\s?|[0-9]{3}-)[0-9]{3}-[0-9]{4}',
    "ordinal": r'[0-9]+-?(th|TH|\'s)',
    "metrics": r'[0-9]+[-A-Za-z]+',
    "money": r'((\$\s*([0-9]+(\.|,|-)*[0-9]*))|(([0-9]+(\.|,|-)*[0-9]*)\s*(\$|((\s-)*(million|billion)('
             r'\s|-)*dollars))))',
    "number": r'[0-9]+(\.|(,(?=[^\s]))|-)*[0-9]*',
    "whitespace": r'(\s|\n|\\|\t)',
    "quote": r'(\"|\')[^\'\"]*(\"|\')',
    "punctuation": r'(-|/|,|(\.\.\.)|\.|:|\?|!|;|\"|\(|\))|(\'(?=[^a-zA-Z]))',
    "email": r'([-_A-Za-z0-9\.])+@[A-Za-z0-9-]+(\.[A-Za-z]+)',
    "website": r'[-A-Za-z0-9]+\.(com|us|uk)',
    "word": r'[A-Za-z\'-]+',
    "unicode_decimal_codes": r'[&#]+[a-z0-9]*;[a-zA-Z0-9-]*',
    "undefined": r'\S+'
}


def annotate_text(text, lang=None):
    if lang is None:
        lang = "english"

    annotations = []
    tokens = tokenize(text=text, patterns=patterns)

    for token in tokens:
        annotations.append(
            Annotation(
                token=token.token,
                token_type=token.token_type,
                stem=stem(word=token.token, lang=lang),
                lemma=lemmatize(word=token.token, text=text)[0]
            )
        )

    return annotations
