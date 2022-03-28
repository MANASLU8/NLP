from pandas import DataFrame

from task1.token_tag import TokenTag
from task2.qwerty_weighting import get_qwerty_weighted_substitution_cost
from task2.sequence_alignment import get_edit_distance
from task3.token_dictionary import load_token_dictionary

_DEBUG_PRINTING = False

CORRECTABLE_TOKEN_TAGS = {TokenTag.PGP_BEGINNING, TokenTag.WORD, TokenTag.ABBREVIATION, TokenTag.PERSON_NAME}
_PREFIX_SIZE = _SUFFIX_SIZE = 2
_TOKEN_CORRECTION_EDIT_DISTANCE_THRESHOLD = 1.9
_TRUSTED_TOKENS_COUNT_THRESHOLD = 50

_dictionary = load_token_dictionary(filter_stopwords=False)
_dictionary_df = _dictionary.df
_dictionary_df["prefix"] = _dictionary_df.token.str[:_PREFIX_SIZE]
_dictionary_df["suffix"] = _dictionary_df.token.str[-_SUFFIX_SIZE:]
_tokens_by_prefix = _dictionary_df.groupby("prefix").token.agg(set)
_tokens_by_suffix = _dictionary_df.groupby("suffix").token.agg(set)


def _get_token_correction(token: str):
    if token in _dictionary.tokens and _dictionary.counts_by_token[token] > _TRUSTED_TOKENS_COUNT_THRESHOLD:
        # optimize correcting known trusted tokens from dictionary
        return token

    if token.isupper() and len(token) < 5:
        # skip correcting abbreviations
        return token

    prefix = token[:_PREFIX_SIZE]
    suffix = token[-_SUFFIX_SIZE:]
    candidates = list(_tokens_by_prefix.get(prefix, set()) | _tokens_by_suffix.get(suffix, set()))
    min_score = None
    min_distance_candidate = None

    for candidate in candidates:
        ed = get_edit_distance(candidate, token, substitution_cost_evaluator=get_qwerty_weighted_substitution_cost)
        if ed > _TOKEN_CORRECTION_EDIT_DISTANCE_THRESHOLD:
            continue
        score = (ed, -_dictionary.counts_by_token[candidate])
        if min_score is None or score < min_score:
            min_score = score
            min_distance_candidate = candidate

    # if we fixed only the case, skip it
    if min_distance_candidate and min_distance_candidate.upper() == token.upper():
        return token

    if min_distance_candidate != token and min_distance_candidate:
        if _DEBUG_PRINTING:
            print(f"{token} -> {min_distance_candidate}")

    return min_distance_candidate or token


def correct_misspelled_tokens(tokens_df: DataFrame):
    fixed_tokens = [
        (_get_token_correction(token) if tag in CORRECTABLE_TOKEN_TAGS else token, tag)
        for token, tag in tokens_df.values
    ]

    return DataFrame(fixed_tokens, columns=["token", "tag"])
