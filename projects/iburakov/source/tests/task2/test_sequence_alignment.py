from faker import Faker
from pytest import mark

from task2.qwerty_weighting import get_qwerty_weighted_substitution_cost
from task2.sequence_alignment import _get_last_alignment_matrix_column, \
    get_optimal_alignment_hirschberg, alignment_to_strings, get_alignment_and_wagner_fischer_matrix

_faker = Faker(use_weighting=False)
_faker.seed_instance(42)


@mark.parametrize(
    "src,dest",
    [
        ("", ""),
        ("", "abc"),
        ("abc", ""),
        ("abc", "abc"),
    ] + [
        (_faker.word(), _faker.word())
        for _ in range(20)
    ]
)
def test_efficient_alignment_matrix_last_column_is_calculated_correctly(src, dest):
    real_last_col = get_alignment_and_wagner_fischer_matrix(src, dest)[0].iloc[:, -1].values
    efficient_last_col = _get_last_alignment_matrix_column(src, dest)
    assert (real_last_col == efficient_last_col).all()


@mark.parametrize("src,dest,expected_result", [
    ("", "", "\n"),
    ("src", "", "src\n"
                "---"),
    ("", "dest", "----\n"
                 "dest"),
    ("prefix", "pre", "prefix\n"
                      "pre---"),
    ("0suffix", "fix", "0suffix\n"
                       "----fix"),
    ("collaboration", "borat", "collaboration\n"
                               "-----borat---"),
    ("collaboration", "boris", "collaboration\n"
                               "-----bor--i-s"),
    ("King Arthur of Camelot", "The Ping Pong Morthu of Spamalot", "K----ing A------rthur of C-amelot\n"
                                                                   "The Ping Pong Morthu- of Spamalot"),
])
def test_optimal_alignment_hirschberg_works(src, dest, expected_result):
    assert "\n".join(alignment_to_strings(get_optimal_alignment_hirschberg(src, dest))) == expected_result


def test_alignment_of_generic_items_works():
    src = [1, 2, 3, 4]
    dest = [2, 33, 4, 5]
    alignment = get_optimal_alignment_hirschberg(src, dest)
    assert alignment == [(1, "-"), (2, 2), (3, 33), (4, 4), ("-", 5)]




