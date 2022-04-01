import re
import enum
from .token_info import TokenInfo
from .lemmatizer import wordnet_lemmatize
from .stemmer import create_stem


class TokenTag(enum.Enum):
    word = "word"
    whitespace = "whitespace"
    punctuation = "punctuation sign"
    name = "name or organization"
    website = "website"
    email = "email"
    phone = "phone number"
    money = "money"
    quote = "quotation"
    number = "number"
    ordinal = "ordinal number"
    metrics = "metrics"
    abbreviations = "abbreviations and cuts"
    unicode_decimal_codes = "word with unicode decimal code"
    undefined = "undefined token type"


patterns = {
    TokenTag.abbreviations: r'(?i)(((calif|ky|inc|corp|st|dr|tel|no|ltd|co|mr|mrs|etc|all)\.)|(u\.s\.|u\.k\.|e\.g\.?)|((?<=\s)[a-z]{1,5}\.(?=\s[a-z])))',
    TokenTag.phone: r'(\([0-9]{3}\)\s?|[0-9]{3}-)[0-9]{3}-[0-9]{4}',
    TokenTag.ordinal: r'[0-9]+-?(th|TH|\'s)',
    TokenTag.metrics: r'[0-9]+[-A-Za-z]+',
    TokenTag.money: r'((\$\s*([0-9]+(\.|,|-)*[0-9]*))|(([0-9]+(\.|,|-)*[0-9]*)\s*(\$|((\s-)*(million|billion)(\s|-)*dollars))))',
    TokenTag.number: r'[0-9]+(\.|(,(?=[^\s]))|-)*[0-9]*',
    TokenTag.whitespace: r'(\s|\n|\\|\t)',
    TokenTag.quote: r'(\"|\')[^\'\"]*(\"|\')',
    TokenTag.punctuation: r'([-/,\.:(\.\.\.)\?!;\"\(\)])|(\'(?=[^a-zA-Z]))',
    TokenTag.email: r'([-_A-Za-z0-9\.])+@[A-Za-z0-9-]+(\.[A-Za-z]+)',
    TokenTag.website: r'[-A-Za-z0-9]+\.(com|us|uk)',
    # TokenTag.name: r'[A-Z][a-z]+\s*[A-Z][a-z]*(?=[\s.])',
    TokenTag.word: r'[A-Za-z\'-]+',
    TokenTag.unicode_decimal_codes: r'[&#]+[a-z0-9]*;[a-zA-Z0-9-]*',
    TokenTag.undefined: r'\S+'
}


def tokenize_news_text(line):
    text_parts = line.split('","')
    news_header_tokens = tokenize(text_parts[1] + ".")
    news_text_tokens = tokenize(text_parts[2][:-2])
    tokens = news_header_tokens + news_text_tokens
    return tokens


def tokenize(text):
    tokens_list = []
    text_to_tokenize = text
    while len(text_to_tokenize) > 0:
        for token_type in patterns:
            match = re.search(r'^(' + patterns[token_type] + ')', text_to_tokenize)
            if match:
                token = match.group()
                lemmatization_result = wordnet_lemmatize(token, text)
                lemma = lemmatization_result[0]
                pos_tag = lemmatization_result[1]
                tokens_list.append(
                    TokenInfo(
                        token,
                        create_stem(token),
                        lemma,
                        pos_tag,
                        token_type.value))
                text_to_tokenize = text_to_tokenize[(len(match.group())):]
                break
            # else:
            # print("No match found")
    # print([str(v) for v in tokens_list])
    return tokens_list
