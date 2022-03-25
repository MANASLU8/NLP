from numpy import diag, ones
from pandas import read_json

from dirs import misc_dir

# Distances generated with code from
# Samuelsson, Axel. "Weighting Edit Distance to Improve Spelling Correction in Music Entity Search." (2017).
_df = read_json(misc_dir / "qwerty_distances.json")
_df = _df - diag(ones(_df.shape[0], dtype=int))
_df = _df.clip(upper=3)
_df = _df ** 0.5
_df = _df / _df.max()
_qwerty_weights = _df.to_dict()


def get_qwerty_weighted_substitution_cost(src_char: str, dest_char: str) -> float:
    if src_char not in _qwerty_weights:
        return 1
    return _qwerty_weights[src_char].get(dest_char, 1)
