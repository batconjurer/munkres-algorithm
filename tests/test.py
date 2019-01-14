import pytest
from _munkres import *
import numpy as np

@pytest.fixture
def matrix_for_tests():
    return np.array([[5, 0, 2, 0], [1, 3, 4, 0], [2, 2, 0, 2]])


def test_matrix_pattern(matrix_for_tests):
    """
    Test that on initialization, the matrix that encodes non-zero entries is
    compute correctly
    """
    munkres_matrix = MunkresMatrix(matrix_for_tests)

    matrix_pattern = [[1, 0, 1, 0], [1, 1, 1, 0], [1, 1, 0, 1]]
    for i in range(3):
        assert list(munkres_matrix.matrix_pattern[i]) == matrix_pattern[i]
    assert munkres_matrix.matrix_pattern.shape == (3, 4)


def test_munkres_initialization(matrix_for_tests):
    """Test that on initialization, all entries of cost matrix are negated"""
    munkres = Munkres(matrix_for_tests)
    matrix = [[-5, 0, -2, 0], [-1, -3, -4, 0], [-2, -2, 0, -2]]
    for i in range(3):
        assert list(munkres.matrix[i]) == matrix[i]
    assert munkres.matrix.shape == (3, 4)


def test_maximal_matching_matrix_adjustment(matrix_for_tests):
    """
    Test that _maximal_matching method correctly subtracts the smallest element of
    each row from every element in the same row
    """
    munkres = Munkres(matrix_for_tests)
    munkres._maximal_matching()
    matrix = [[0, 5, 3, 5], [3, 1, 0, 4], [0, 0, 2, 0]]

    for i in range(3):
        assert list(munkres.matrix[i]) == matrix[i]

    assert munkres.matrix.shape == (3, 4)


def test_maximal_matching_marked(matrix_for_tests):
    """
    Test that the matrix encoding the entries of a maximal
    matching of the 0-induced matrix are computed correctly
    """
    munkres = Munkres(matrix_for_tests)
    munkres._maximal_matching()
    marked = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]

    for i in range(3):
        assert list(munkres.marked[i]) == marked[i]

    assert munkres.marked.shape == (3, 4)


def test_row_and_col_saturated_after_maximal_matching(matrix_for_tests):
    """
    Test that the computation of a maximal matching for the 0-induced graph
    correctly labels the appropriate row and column vertices as saturated
    """
    munkres = Munkres(matrix_for_tests)
    munkres._maximal_matching()
    assert list(munkres.row_saturated) == [True, True, True]
    assert list(munkres.col_saturated) == [True, False, True, True]


def test_remove_covers(matrix_for_tests):
    """
    Test that the remove covers function resets all appropriate vectors to have all entries
    False and that marked contain only zeros
    """
    munkres = Munkres(matrix_for_tests)
    munkres.col_saturated += 1
    munkres.row_saturated += 1
    munkres.marked += 1
    munkres.row_marked += 1
    munkres.col_marked += 1
    assert np.all(munkres.marked)
    assert munkres.marked.shape == (3, 4)
    assert list(munkres.col_saturated) == [True, True, True, True]
    assert list(munkres.row_saturated) == [True, True, True]
    assert list(munkres.col_marked) == [True, True, True, True]
    assert list(munkres.row_marked) == [True, True, True]
    munkres._remove_covers()
    assert not np.any(munkres.marked)
    assert munkres.marked.shape == (3, 4)
    assert list(munkres.col_saturated) == [False, False, False, False]
    assert list(munkres.row_saturated) == [False, False, False]
    assert list(munkres.col_marked) == [False, False, False, False]
    assert list(munkres.row_marked) == [False, False, False]



"""
def test_max_weight_matching(matrix_for_tests):
    # Fully saturated case, wide
    munk_one = _get_matrix(matrix_for_tests)
    # Not saturated case, wide
    munk_two = _get_matrix(np.array([[5, 0, 2, 0], [1, 3, 4, 0], [2, 0, 0, 0]], dtype=float))
    # Not saturated case, tall
    munk_three = _get_matrix(matrix_for_tests, [3, 1, 1])
    # Saturated case tall
    munk_four = _get_matrix(matrix_for_tests, [3, 1, 2])

    assert set(munk_one.maximum_weight_matching()) == {(0, 0), (1, 2), (2, 1)}
    assert set(munk_two.maximum_weight_matching()) == {(0, 0), (1, 2)}
    assert set(munk_three.maximum_weight_matching()) == {(0, 0), (0, 2), (1, 1), (2, 3)}
    assert set(munk_four.maximum_weight_matching()) == {(0, 0), (1, 2), (2, 1), (2, 3)}
"""