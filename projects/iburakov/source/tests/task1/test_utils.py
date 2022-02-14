import re
from typing import List, Tuple

from pytest import mark

from task1.utils import split_keep_delimiter_and_metadata


@mark.parametrize("delimiter_re,s,expected_result", [
    (r"", "zzz", [(False, "z"), (False, "z"), (False, "z")]),
    (r"a\wb", "zzz", [(False, "zzz")]),
    (r"a\wb", "aob", [(True, "aob")]),
    (r"a\wb", "baob", [(False, "b"), (True, "aob")]),
    (r"a\wb", "aobab", [(True, "aob"), (False, "ab")]),
    (r"a\wb", "baobab", [(False, "b"), (True, "aob"), (False, "ab")]),
])
def test_split_keep_delimiter_and_metadata_works(delimiter_re: str, s: str, expected_result: List[Tuple[bool, str]]):
    assert list(split_keep_delimiter_and_metadata(re.compile(delimiter_re), s)) == expected_result
