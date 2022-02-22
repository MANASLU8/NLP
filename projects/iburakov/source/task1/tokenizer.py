import re
from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional

from pandas import DataFrame

from task1.token_tag import TokenTag
from task1.utils import split_keep_delimiter_and_metadata

_word = r"([a-zA-Z\d']+(-[a-zA-Z\d']+)?)"
_number = r"((\d{1,3}[ ,.])?(\d{3}[ ,.])*\.?\d+(\.\d+)?([kmg]|st|nd|rd|th)?(\s*\d*/\d+)?)"
_currencies = r"(\$|£|€|¥|฿|US\s?\$|USD|C\s?\$|CAD|AUD|₽|RUB|RUR)"
_currency_sep = r"\s*"

_url = (
    r"((?=[\w])(?:(?:https?|ftp|mailto)://)?(?:\S+(?::\S*)?@)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)"
    r"(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2"
    r"}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z0-9\-]*)?[a-z0-9]+)(?:\.(?:[a-z0-9])(?"
    r":[a-z0-9\-])*[a-z0-9])?(?:\.(?:[a-z]{2,})))(?::\d{2,5})?(?:/\S*)?\??(:?\S*)?(?<=[\w/]))"
)  # based on spaCy's regex

_emoticons = "(" + "|".join(re.escape(
    ":),:-),:)),:-)),:))),:-))),(:,(-:,=),(=,:],:-],[:,[-:,[=,=],:o),(o:,:},:-},8),8-),(-8,;),;-),(;,(-;,:(,:-(,:((,"
    ":-((,:(((,:-(((,):,)-:,=(,>:(,:'),:'-),:'(,:'-(,:/,:-/,=/,=|,:|,:-|,]=,=[,:P,:-P,:p,:-p,:O,:-O,:o,:-o,:-0,:(),>:o,"
    ":*,:-*,:-3,=3,:>,:->,:X,:-X,:x,:-x,:D,:-D,;D,;-D,=D,xD,XD,xDD,XDD,8D,8-D,^_^,^__^,^___^,>.<,>.>,<.<,._.,;_;,-_-,"
    "-__-,v.v,V.V,v_v,V_V,o_o,o_O,O_o,O_O,0_o,o_0,0_0,o.O,O.o,O.O,o.o,0.0,o.0,0.o,@_@,<3,<33,<333,</3,(^_^),(-_-),"
    "(._.),(>_<),(*_*),(¬_¬),ಠ_ಠ,ಠ︵ಠ,(ಠ_ಠ),¯\(ツ)/¯,:0,:1,:3"
).split(",")) + ")"  # based on spaCy's emoticons

# based on https://github.com/stanfordnlp/CoreNLP/blob/f05cb54ec0a4f3c90395771817f44a81eb549baf/src/edu/stanford/nlp/process/PTBLexer.flex#L690
_abbreviations = (
    r"("
    r"inc|co|corp|ltd|plc|rt|bancorp|bhd|assn|univ|intl|sys|tel|est|ext|sq|jr|sr|bros|(Ed|Ph)\.D|[BDM]\.Sc|LL\.[BDM]|"
    r"Sci|etc|ect|al|seq|orig|incl|eg|avg|pl|lb|min|max|cit|u\.s|u\.k|Mr|Mrs|Ms|Mx|[M]iss|Drs?|Profs?|Sens?|Reps?|"
    r"Attys?|Lt|Col|Gen|Messrs|Govs?|Adm|Rev|Fr|Rt|Maj|Sgt|Cpl|Pvt|Capt|St[ae]?|Ave|Pres|Lieut|Rt|Hon|Brig|Co?mdr|Pfc|"
    r"Spc|Supts?|Det|Mt|Ft|Adj|Adv|Asst|Assoc|Ens|Insp|Mlle|Mme|Msgr|Sfc|Amb|S[m][t]|Ven|Br|EngInvt|Elec|Natl|M[ft]g|"
    r"Dept|Blvd|Rd|Ave|[P][l]|viz|Exhs?|ass't|Govt|Wm|Jos|Cie|cf|TREAS|P[h]|[S][c]|syn|def|Mk|Soc|vs|a\.k\.a|sect?s?|"
    r"prop|op|approx|pt|i\.e|e\.g"
    r")"
)

_case_dependant_tags = {TokenTag.PERSON_NAME}


def _get_re_flags(tag: str):
    return re.DOTALL | re.IGNORECASE if tag not in _case_dependant_tags else re.DOTALL


# tag -> regexp, sorted by precedence
_TOKEN_MATCHERS = {k: re.compile(v, _get_re_flags(k)) for k, v in {
    TokenTag.QUOTE_HEADER: rf"(In (article )?(<{_url}>,?\s*)?)?{_url}\s*(\((\s*({_word}\.?|-+)){{1,8}}\s*\) )?writes",
    TokenTag.PGP_BEGINNING: r"-----BEGIN PGP SIGNED MESSAGE-----",
    TokenTag.PGP_PUBLIC_KEY: r"-----BEGIN PGP PUBLIC KEY BLOCK-----.*?-----END PGP PUBLIC KEY BLOCK-----",
    TokenTag.PGP_SIGNATURE: r"-----BEGIN PGP SIGNATURE-----.*?-----END PGP SIGNATURE-----",
    TokenTag.PERSON_NAME: r"([A-Z]\.\s?([A-Z]([a-z]+|\.)\s?)?[A-Z][a-z]+)",
    TokenTag.URL_EMAIL: _url,
    TokenTag.MONEY: rf"{_number}{_currency_sep}{_currencies}|{_currencies}{_currency_sep}{_number}",
    TokenTag.PHONE_NUMBER: r"(\(\+?\d+\)|\+|\d+-)[\- ]*(\(?\d+\)?[\- ]?)+(\s*(ext\.|/)\s*\d+)?",
    TokenTag.TIME: r"([01]?\d|20|21|22|23)(\s*(am|pm)|:[0-5]\d(\s*(am|pm)|:[0-5]\d)?)",
    TokenTag.EMOTICON: _emoticons,
    TokenTag.NUMBER: rf"(?<!\w){_number}(?!\w)",
    TokenTag.PUNCT_COMMA: r",",
    TokenTag.ABBREVIATION: rf"(?<!\w){_abbreviations}\.",
    TokenTag.PUNCT_SENTENCE: r"\.{1,3}|[!?]+",
    TokenTag.PUNCT_BRACES: r"[()]",
    TokenTag.PUNCT_QUOTES: r"[\"”“`‘´’‚,„»«]",
    TokenTag.PUNCT_COLON: r"[:]",
    TokenTag.WORD: _word,
}.items()}


@dataclass
class _TaggedToken:
    text: str
    tag: Optional[str] = None


def tokenize_text(text: str, print_times: bool = False) -> DataFrame:
    tokens: List[_TaggedToken] = [_TaggedToken(text)]
    for tag, matcher in _TOKEN_MATCHERS.items():
        start_time = perf_counter()

        new_tokens = []
        for token in tokens:
            if token.tag is not None:
                new_tokens.append(token)
            else:
                new_tokens.extend((
                    _TaggedToken(text=part, tag=tag if is_match else None)
                    for is_match, part in split_keep_delimiter_and_metadata(matcher, token.text)
                ))
        tokens = new_tokens

        if print_times:
            print(tag, f"{(perf_counter() - start_time) * 1000:.3f} ms")

    return DataFrame(({"token": token.text.strip(), "tag": token.tag} for token in tokens if token.tag is not None))
