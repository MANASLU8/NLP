from pytest import mark

from task2.qwerty_weighting import get_qwerty_weighted_substitution_cost
from task2.sequence_alignment import alignment_to_strings, get_optimal_alignment_hirschberg, get_edit_distance


def test_qwerty_weighted_substitution_cost_is_calculated_correctly():
    assert get_qwerty_weighted_substitution_cost("a", "a") == 0
    assert 0.7 > get_qwerty_weighted_substitution_cost("a", "z") > 0
    assert get_qwerty_weighted_substitution_cost("z", "p") == 1


def test_weighted_editing_distance_is_correct():
    assert get_edit_distance("a", "s", get_qwerty_weighted_substitution_cost) < 1


_QWERTY_TEST_SRC = "bbbb qwe uio bbbb"


@mark.parametrize("dest,expected_dest_with_gaps", [
    ("asd", "-----asd---------"),
    ("jkl", "---------jkl-----"),
])
def test_qwerty_weighted_string_alignment_works(dest, expected_dest_with_gaps):
    _, dest_with_gaps = alignment_to_strings(get_optimal_alignment_hirschberg(
        _QWERTY_TEST_SRC, dest, get_qwerty_weighted_substitution_cost
    ))
    assert dest_with_gaps == expected_dest_with_gaps
