from numbers import Number
from typing import TypeVar, Sequence, List, Tuple, Callable

from numpy import zeros, ndarray, arange, array
from pandas import DataFrame

TItem = TypeVar("TItem")
Alignment = List[Tuple[str, str]]
CostEvaluator = Callable[[TItem, TItem], Number]

_INSERTION_COST = _DELETION_COST = 1


class _Action:
    INSERTION = "i"
    DELETION = "d"
    SUBSTITUTION = "s"


def _get_generic_substitution_cost(src_item: TItem, dest_item: TItem) -> Number:
    return int(src_item != dest_item)


def _backtrack_alignment_from_actions(actions_df: DataFrame) -> Alignment:
    result = []
    y, x = actions_df.shape
    y -= 1
    x -= 1

    while not (x == y == 0):
        dest_item, src_item = actions_df.index[y], actions_df.columns[x]
        action = actions_df.iloc[y, x]
        if action == _Action.SUBSTITUTION:
            result.append((src_item, dest_item))
            y -= 1
            x -= 1
        elif action == _Action.INSERTION:
            result.append(("-", dest_item))
            y -= 1
        elif action == _Action.DELETION:
            result.append((src_item, "-"))
            x -= 1

    return result[::-1]


def get_alignment_and_wagner_fischer_matrix(
        src: Sequence[TItem], dest: Sequence[TItem],
        substitution_cost_evaluator: CostEvaluator = _get_generic_substitution_cost) -> Tuple[DataFrame, Alignment]:
    df = DataFrame(0, index=("", *dest), columns=("", *src), dtype=float)
    actions_df = df.copy()

    len_y, len_x = df.shape
    df[""] = array(range(len_y), dtype=float) * _INSERTION_COST
    df.loc[""] = array(range(len_x), dtype=float) * _DELETION_COST
    actions_df[""] = _Action.INSERTION
    actions_df.loc[""] = _Action.DELETION

    for x in range(1, len_x):
        for y in range(1, len_y):
            dest_item, src_item = df.index[y], df.columns[x]
            options = {
                _Action.SUBSTITUTION: df.iloc[y - 1, x - 1] + substitution_cost_evaluator(src_item, dest_item),
                _Action.DELETION: df.iloc[y, x - 1] + _DELETION_COST,
                _Action.INSERTION: df.iloc[y - 1, x] + _INSERTION_COST,
            }
            action, value = min(options.items(), key=lambda v: v[1])
            df.iloc[y, x] = value
            actions_df.iloc[y, x] = action

    return df, _backtrack_alignment_from_actions(actions_df)


def _get_last_alignment_matrix_column(
        src: Sequence[TItem], dest: Sequence[TItem],
        substitution_cost_evaluator: CostEvaluator = _get_generic_substitution_cost) -> ndarray:
    """ Computes last column of alignment matrix from src to dest in a space-efficient manner """
    len_y = len(dest) + 1
    prev_col = arange(len_y, dtype=float) * _INSERTION_COST

    for col_index, src_item in enumerate(src):
        cur_col = zeros(shape=len_y, dtype=float)
        # col_index excludes first column, actually, so it's off by 1
        cur_col[0] = _DELETION_COST * (col_index + 1)

        for row_index, dest_item in enumerate(dest):
            i = row_index + 1  # row_index also excludes first matrix row (labeled ""), it's initialized above
            cur_col[i] = min(
                prev_col[i - 1] + substitution_cost_evaluator(src_item, dest_item),
                prev_col[i] + _DELETION_COST,
                cur_col[i - 1] + _INSERTION_COST,
            )

        prev_col = cur_col

    return prev_col


def get_optimal_alignment_hirschberg(src: Sequence[TItem], dest: Sequence[TItem],
                                     substitution_cost_evaluator: CostEvaluator = _get_generic_substitution_cost) \
        -> Alignment:
    if len(dest) == 0:
        return [(c, "-") for c in src]
    if len(src) == 0:
        return [("-", c) for c in dest]
    if len(src) == 1 or len(dest) == 1:
        # gets alignment with matrix backtracking for small sequences
        return get_alignment_and_wagner_fischer_matrix(src, dest, substitution_cost_evaluator)[1]

    x_mid = len(src) // 2
    src_l, src_r = src[:x_mid], src[x_mid:]
    score_l = _get_last_alignment_matrix_column(src_l, dest, substitution_cost_evaluator)
    # reusing forward subroutine to go backward by reversing strings
    score_r = _get_last_alignment_matrix_column(src_r[::-1], dest[::-1], substitution_cost_evaluator)[::-1]
    y_mid = (score_l + score_r).argmin()
    dest_l, dest_r = dest[:y_mid], dest[y_mid:]
    return (get_optimal_alignment_hirschberg(src_l, dest_l, substitution_cost_evaluator)
            + get_optimal_alignment_hirschberg(src_r, dest_r, substitution_cost_evaluator))


def get_edit_distance(src: Sequence[TItem], dest: Sequence[TItem],
                      substitution_cost_evaluator: CostEvaluator = _get_generic_substitution_cost) \
        -> float:
    return _get_last_alignment_matrix_column(src, dest, substitution_cost_evaluator)[-1]


def evaluate_alignment_edit_distance(
        alignment: Alignment, substitution_cost_evaluator: CostEvaluator = _get_generic_substitution_cost) -> float:
    result = 0.0
    for src_item, dest_item in alignment:
        if src_item == "-":
            result += _INSERTION_COST
        elif dest_item == "-":
            result += _DELETION_COST
        else:
            result += substitution_cost_evaluator(src_item, dest_item)

    return result


def alignment_to_strings(alignment: Alignment) -> Tuple[str, str]:
    """ :returns (src_with_gaps, dest_with_gaps) """
    if not alignment:
        return "", ""

    src, dest = ["".join(s) for s in zip(*alignment)]
    return src, dest


def print_alignment(src, dest):
    src, dest = alignment_to_strings(get_optimal_alignment_hirschberg(src, dest))
    print(f"Alignment: \t{src}\n\t\t\t\t{dest}")

# TODO: refactor parametrization with substitution_cost_evaluator (inheritance?)
