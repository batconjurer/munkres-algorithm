import pytest
from _munkres import *
import numpy as np

@pytest.fixture
def matrix_for_tests():
    return np.array([[5, 0, 2, 0], [1, 3, 4, 0], [2, 2, 0, 2]])


def test_munkres_initialization_matrix_negation(matrix_for_tests):
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
        assert list(munkres.marked.astype(int)[i]) == marked[i]

    assert munkres.marked.shape == (3, 4)


def test_row_and_col_saturated_after_maximal_matching(matrix_for_tests):
    """
    Test that the computation of a maximal matching for the 0-induced graph
    correctly labels the appropriate row and column vertices as saturated
    """
    munkres = Munkres(matrix_for_tests)
    munkres._maximal_matching()
    assert list(munkres.row_saturated) == [True, True, True]
    assert list(munkres.col_saturated) == [True, True, True, False]


def test_remove_covers(matrix_for_tests):
    """
    Test that the remove covers function resets all appropriate vectors to have all entries
    False and that marked contain only zeros
    """
    munkres = Munkres(matrix_for_tests)
    munkres.col_saturated += True
    munkres.row_saturated += True
    munkres.marked += True
    munkres.row_marked += True
    munkres.col_marked += True
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


def test_aug_paths_1():

    munkres = Munkres(np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1],
                                [0, 0, 0, 1, 1],
                                [0, 0, 1, 1, 1],
                                [0, 1, 1, 1, 1]], dtype=float))
    munkres.marked = np.array([[1, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]], dtype=bool)
    munkres.row_saturated = np.array([True, True, True, False, False], dtype=bool)
    munkres.col_saturated = np.array([True, True, True, False, False], dtype=bool)

    munkres._aug_paths()
    marked = [[0, 0, 0, 0, 1],
              [0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0]]
    for i in range(5):
        assert list(munkres.marked.astype(int)[i]) == marked[i]
    assert munkres.marked.shape == (5, 5)


def test_aug_paths_2():
    munkres = Munkres(np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1],
                                [0, 0, 0, 1, 1],
                                [0, 0, 1, 1, 1],
                                [1, 1, 1, 1, 1]], dtype=float))
    munkres.marked = np.array([[1, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]], dtype=bool)

    munkres.row_saturated = np.array([True, True, True, False, False], dtype=bool)
    munkres.col_saturated = np.array([True, True, True, False, False], dtype=bool)

    munkres._aug_paths()
    marked = [[0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]
    for i in range(5):
        assert list(munkres.marked.astype(int)[i]) == marked[i]

    assert munkres.marked.shape == (5, 5)


def test_aug_paths_3():
    munkres = Munkres(np.array([[0, 1, 1, 1, 1, 1],
                                [0, 0, 0, 1, 1, 1],
                                [0, 1, 1, 0, 0, 1],
                                [1, 1, 0, 0, 1, 0],
                                [1, 1, 1, 0, 1, 1]], dtype=float))

    munkres.marked = np.array([[0, 0, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]], dtype=bool)

    munkres.row_saturated = np.array([False, True, True, True, False], dtype=bool)
    munkres.col_saturated = np.array([True, False, True, True, False, False], dtype=bool)

    munkres._aug_paths()
    marked = [[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0]]

    for i in range(5):
        assert list(munkres.marked[i].astype(int)) == marked[i]

    assert munkres.marked.shape == (5, 6)


def test_max_weight_matching_1(matrix_for_tests):
    """ Test that the correct maximum weight matching is found"""
    # Fully saturated case, wide
    munkres_one = linear_sum_assignment(matrix_for_tests)
    assert set(munkres_one) == {(0, 0), (1, 2), (2, 1)}


def test_max_weight_matching_2():
    # Not saturated case, wide
    munkres_two = linear_sum_assignment(np.array([[5, 0, 2, 0],
                                                  [1, 3, 4, 0],
                                                  [2, 0, 0, 0]], dtype=float))
    assert set(munkres_two) == {(0, 0), (1, 2), (2, 1)}


def test_max_weight_matching_3():
    # Not saturated case, tall
    munkres_three = linear_sum_assignment(np.array([[5, 0, 2, 0],
                                                    [5, 0, 2, 0],
                                                    [5, 0, 2, 0],
                                                    [1, 3, 4, 0],
                                                    [2, 2, 0, 2]], dtype=float))
    assert set(munkres_three) == {(0, 0), (1, 2), (3, 1), (4, 3)}


def test_max_weight_matching_4():
    # Saturated case tall
    munkres_four = linear_sum_assignment(np.array([[5, 0, 2, 0],
                                                   [5, 0, 2, 0],
                                                   [5, 0, 2, 0],
                                                   [1, 3, 4, 0],
                                                   [2, 2, 0, 2],
                                                   [2, 2, 0, 2]], dtype=float))

    assert set(munkres_four) == {(0, 0), (3, 2), (4, 1), (5, 3)}
