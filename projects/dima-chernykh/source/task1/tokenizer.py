import re
from dataclasses import dataclass
from typing import Optional

from pandas import DataFrame


email = r'[a-zA-Z0-9+_.-]+([%]?[a-zA-Z0-9+_.-])*@[a-zA-Z0-9.-]+([%]?[a-zA-Z0-9+_.-])*'
attribute = r'(?P<attribute>\w+):'
attribute_sent = rf"^{attribute} (?P<value>.*)$"
name = r"[A-Za-z][a-z]*"
word = r"[a-zA-Z']+([-&]?[a-zA-Z']+)?"
underscore_words = rf"_(\s*{word})*_"
asterisks_words = rf"\*(\s*{word})*\*"
number = r"(?<!\w)((\d+[ ,.])?(\d+[ ,.])*\.?\d+(\.\d+)?(\s*\d*/\d+)?)(k|K|m|M|st|nd|th|g|G)?(?!\w)"
version = rf"v{number}(.{number})*"
intro_sent = rf'(In (article )?)?.*writes'
sentence_end = r"\.{1,3}|[!?]+"
punc_colon = r'\:'
punc_brace = r"[()]"
punc_comma = r','
punc_eq = r'='
punc_quote = r'[\"«»”“„`´‘’‚,]'
emotion = "(" + "|".join(re.escape(
    ":),:-),:)),:-)),:))),:-))),(:,(-:,=),(=,:],:-],[:,[-:,[=,=],:o),(o:,:},:-},8),8-),(-8,;),;-),(;,(-;,:(,:-(,:((,"
    ":-((,:((("
).split(",")) + ")"
pgp_signature = r"-----BEGIN PGP SIGNATURE-----.*?-----END PGP SIGNATURE-----"
pgp_key = r"-----BEGIN PGP PUBLIC KEY BLOCK-----.*?-----END PGP PUBLIC KEY BLOCK-----"
pgp_signed_header = r"-----BEGIN PGP SIGNED MESSAGE-----"
money = rf"{number}\s*(\$|USD|€|£|฿)|(\$|USD|€|£|฿)\s*{number}"
phone_num = r"[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}"
time = r"[0-9]{1,2}:[0-9]{1,2}\s*(am|pm|AM|PM)?"
month = "(" + "|".join(re.escape(
    "January,Jan,February,Feb,March,Mar,April,Apr,May,June,Jun,July,Jul,August,Aug,September,Sep,October,Oct,"
    "November,Nov,December,Dec"
).split(",")) + ")"
web_sait = r"(http:|https:)//(.*(/.*)*)"


tag_attr_token_matcher = {k: re.compile(v, re.DOTALL) for k, v in {
    "ATTR": attribute,
    "EMAIL": email,
    "NUM": number,
    "PUNC_COLON": punc_colon,
    "PUNC_BRACE": punc_brace,
    "PUNC_COMMA": punc_comma,
    "PUNC_QUOTE": punc_quote,
    "WORD": word,
}.items()}

tag_intro_sent_matcher = {k: re.compile(v, re.DOTALL) for k, v in {
    "EMAIL": email,
    "NUM": number,
    "PUNC_COLON": punc_colon,
    "PUNC_BRACE": punc_brace,
    "PUNC_COMMA": punc_comma,
    "PUNC_QUOTE": punc_quote,
    "WORD": word,
}.items()}

tag_text_token_matcher = {k: re.compile(v, re.DOTALL) for k, v in {
    "PGP_SIGNED_BEGIN": pgp_signed_header,
    "PGP_KEY": pgp_key,
    "PGP_SIG": pgp_signature,
    "INTRO_SENT": intro_sent,
    "EMAIL": email,
    "WEB": web_sait,
    "MONTH": month,
    "MONEY": money,
    "PHONE_NUM": phone_num,
    "TIME": time,
    "EMOT": emotion,
    "NUM": number,
    "PUNC_COMMA": punc_comma,
    "SENT_END": sentence_end,
    "PUNC_BRACE": punc_brace,
    "PUNC_QUOTE": punc_quote,
    "PUNC_COLON": punc_colon,
    "WORD": word,
}.items()}


@dataclass
class MetaToken:
    token: str
    tag: Optional[str] = None


def parse_tokens(text, tag_matcher):
    tokens = [MetaToken(text)]
    for tag, matcher in tag_matcher.items():
        temp_tokens = []
        for m_token in tokens:
            if m_token.tag is not None:
                temp_tokens.append(m_token)
            else:
                end_pos = 0
                for match in matcher.finditer(m_token.token):
                    first_pos, last_pos = match.span()
                    if end_pos != first_pos:
                        temp_tokens.append(MetaToken(token=m_token.token[end_pos:first_pos]))
                    if first_pos != last_pos:
                        if tag == "INTRO_SENT":
                            temp_tokens.extend(parse_tokens(m_token.token[first_pos:last_pos], tag_intro_sent_matcher))
                            temp_tokens.append(MetaToken(token="FAKE_END", tag="SENT_END"))
                        else:
                            temp_tokens.append(MetaToken(token=m_token.token[first_pos:last_pos], tag=tag))
                    end_pos = last_pos
                if end_pos != len(m_token.token):
                    temp_tokens.append(MetaToken(token=m_token.token[end_pos:]))
        tokens = temp_tokens
    return tokens


def tokenize(text):
    try:
        attribute_sentences, other_text = text.split("\n\n", maxsplit=1)
        meta_tokens = parse_tokens(attribute_sentences, tag_attr_token_matcher)
        meta_tokens.append(MetaToken(token="FAKE_END", tag="SENT_END"))
        meta_tokens.extend(parse_tokens(other_text, tag_text_token_matcher))
    except ValueError:
        meta_tokens = parse_tokens(text, tag_text_token_matcher)
    return DataFrame(({"token": tk.token.strip(), "tag": tk.tag} for tk in meta_tokens if tk.tag is not None))


if __name__ == "__main__":
    pass
