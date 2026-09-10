"""Pure-function regression tests for CN_format / process_blimp_score -- no
model or file I/O needed for these two."""

from baby_logic_lm.evaluation.evaluate import CN_format, process_blimp_score


def test_cn_format_ranks_lowest_nll_first():
    # Two minimal sets of 3 candidates each; lower NLL = more preferred.
    scores_log = [
        [3.0, 1.0, 2.0],  # position 1 wins (rank 1), position 2 rank 2, position 0 rank 3
        [1.0, 2.0, 3.0],  # position 0 wins
    ]
    result = CN_format(scores_log)

    assert result[0] == {3: 1, 1: 1}
    assert result[1] == {1: 1, 2: 1}
    assert result[2] == {2: 1, 3: 1}


def test_process_blimp_score_fraction_correct():
    pairs = [(1.0, 2.0), (2.0, 1.0), (1.0, 5.0), (5.0, 1.0)]
    # good < bad for pairs[0] and pairs[2] -> 2/4 correct
    assert process_blimp_score(pairs) == 0.5


def test_process_blimp_score_all_correct():
    pairs = [(1.0, 2.0), (0.5, 3.0)]
    assert process_blimp_score(pairs) == 1.0
