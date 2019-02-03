import pytest
from _munkres import *
import numpy as np
from numpy.testing import assert_array_equal

@pytest.fixture
def matrix_for_tests():
    return np.array([[5, 0, 2, 0], [1, 3, 4, 0], [2, 2, 0, 2]])


def Munkres(matrix, transposed=False, maximize=False):
    if maximize:
        return State(marked=np.zeros(matrix.shape, dtype=bool),
                     row_saturated=np.zeros(matrix.shape[0], dtype=bool),
                     col_saturated=np.zeros(matrix.shape[1], dtype=bool),
                     row_marked=np.zeros(matrix.shape[0], dtype=bool),
                     col_marked=np.zeros(matrix.shape[1], dtype=bool),
                     matrix=-matrix)
    else:
        return State(marked=np.zeros(matrix.shape, dtype=bool),
                     row_saturated=np.zeros(matrix.shape[0], dtype=bool),
                     col_saturated=np.zeros(matrix.shape[1], dtype=bool),
                     row_marked=np.zeros(matrix.shape[0], dtype=bool),
                     col_marked=np.zeros(matrix.shape[1], dtype=bool),
                     matrix=matrix)


def test_maximal_matching_matrix_adjustment_minimize_option(matrix_for_tests):
    """
    Test that _maximal_matching method correctly subtracts the smallest element of
    each row from every element in the same row
    """
    state = Munkres(-matrix_for_tests)
    maximal_matching(state)
    matrix = np.array([[0, 5, 3, 5], [3, 1, 0, 4], [0, 0, 2, 0]], dtype=float)
    assert_array_equal(state.matrix, matrix)


def test_maximal_matching_matrix_adjustment_maximize_option(matrix_for_tests):
    """
    Test that _maximal_matching method correctly subtracts the smallest element of
    each row from every element in the same row
    """
    state = Munkres(matrix_for_tests, maximize=True)
    maximal_matching(state)
    matrix = np.array([[0, 5, 3, 5], [3, 1, 0, 4], [0, 0, 2, 0]], dtype=float)
    assert_array_equal(state.matrix, matrix)


def test_maximal_matching_marked_minimize_option(matrix_for_tests):
    """
    Test that the matrix encoding the entries of a maximal
    matching of the 0-induced matrix are computed correctly
    """
    state = Munkres(-matrix_for_tests)
    maximal_matching(state)
    marked = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=bool)
    assert_array_equal(state.marked, marked)


def test_maximal_matching_marked_maximize_option(matrix_for_tests):
    """
    Test that the matrix encoding the entries of a maximal
    matching of the 0-induced matrix are computed correctly
    """
    state = Munkres(matrix_for_tests, maximize=True)
    maximal_matching(state)
    marked = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=bool)
    assert_array_equal(state.marked, marked)


def test_row_and_col_saturated_after_maximal_matching_minimize_option(matrix_for_tests):
    """
    Test that the computation of a maximal matching for the 0-induced graph
    correctly labels the appropriate row and column vertices as saturated
    """
    state = Munkres(-matrix_for_tests)
    maximal_matching(state)
    assert_array_equal(state.row_saturated, np.array([True, True, True], dtype=bool))
    assert_array_equal(state.col_saturated, np.array([True, True, True, False], dtype=bool))


def test_row_and_col_saturated_after_maximal_matching_maximize_option(matrix_for_tests):
    """
    Test that the computation of a maximal matching for the 0-induced graph
    correctly labels the appropriate row and column vertices as saturated
    """
    state = Munkres(matrix_for_tests, maximize=True)
    maximal_matching(state)
    assert_array_equal(state.row_saturated, np.array([True, True, True], dtype=bool))
    assert_array_equal(state.col_saturated, np.array([True, True, True, False], dtype=bool))


def test_remove_covers(matrix_for_tests):
    """
    Test that the remove covers function resets all appropriate vectors to have all entries
    False and that marked contain only zeros
    """
    state = Munkres(matrix_for_tests)
    state.col_saturated += True
    state.row_saturated += True
    state.marked += True
    state.row_marked += True
    state.col_marked += True

    assert_array_equal(state.marked, np.ones((3, 4), dtype=bool))
    assert_array_equal(state.col_saturated, np.array([True, True, True, True],dtype=bool))
    assert_array_equal(state.row_saturated, np.array([True, True, True], dtype=bool))
    assert_array_equal(state.col_marked, np.array([True, True, True, True], dtype=bool))
    assert_array_equal(state.row_marked, np.array([True, True, True], dtype=bool))

    remove_covers(state)
    assert_array_equal(state.marked, np.zeros((3, 4), dtype=bool))
    assert_array_equal(state.col_saturated, np.array([False, False, False, False], dtype=bool))
    assert_array_equal(state.row_saturated, np.array([False, False, False], dtype=bool))
    assert_array_equal(state.col_marked, np.array([False, False, False, False], dtype=bool))
    assert_array_equal(state.row_marked, np.array([False, False, False], dtype=bool))


def test_aug_paths_1():
    """ Test the algorithm that finds a maximum matching from a maximal matching via augmenting
        paths
    """
    # Original biadjacency matrix where zeros represent an edge and non-zero values represent
    # non-edges
    state = Munkres(np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 1, 1],
                              [0, 0, 1, 1, 1],
                              [0, 1, 1, 1, 1]], dtype=float))
    # A maximal matching that is not maximum
    state.marked = np.array([[1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]], dtype=bool)
    state.row_saturated = np.array([True, True, True, False, False], dtype=bool)
    state.col_saturated = np.array([True, True, True, False, False], dtype=bool)

    aug_paths(state)
    # The resulting (unique) maximum matching
    marked = np.array([[0, 0, 0, 0, 1],
                       [0, 0, 0, 1, 0],
                       [0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0]], dtype=bool)
    assert_array_equal(state.marked, marked)


def test_aug_paths_2():
    """ Test the algorithm that finds a maximum matching from a maximal matching via augmenting
            paths
    """
    # Original biadjacency matrix where zeros represent an edge and non-zero values represent
    # non-edges
    state = Munkres(np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 1, 1],
                              [0, 0, 1, 1, 1],
                              [1, 1, 1, 1, 1]], dtype=float))
    # A maximal matching that is not maximum
    state.marked = np.array([[1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]], dtype=bool)

    state.row_saturated = np.array([True, True, True, False, False], dtype=bool)
    state.col_saturated = np.array([True, True, True, False, False], dtype=bool)

    aug_paths(state)
    # The resulting maximum matching
    marked = np.array([[0, 0, 0, 1, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]], dtype=bool)
    assert_array_equal(state.marked, marked)


def test_aug_paths_3():
    """ Test the algorithm that finds a maximum matching from a maximal matching via augmenting
            paths
    """
    # Original biadjacency matrix where zeros represent an edge and non-zero values represent
    # non-edges
    state = Munkres(np.array([[0, 1, 1, 1, 1, 1],
                              [0, 0, 0, 1, 1, 1],
                              [0, 1, 1, 0, 0, 1],
                              [1, 1, 0, 0, 1, 0],
                              [1, 1, 1, 0, 1, 1]], dtype=float))
    # A maximal matching that is not maximum
    state.marked = np.array([[0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]], dtype=bool)

    state.row_saturated = np.array([False, True, True, True, False], dtype=bool)
    state.col_saturated = np.array([True, False, True, True, False, False], dtype=bool)

    aug_paths(state)
    # The resulting maximum matching
    marked = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0]], dtype=bool)

    assert_array_equal(state.marked, marked)


def test_min_weight_matching_1(matrix_for_tests):
    """ Test that the correct minimum weight matching is found"""
    # Fully saturated case, wide
    munkres_one = linear_sum_assignment(-matrix_for_tests)
    row_result = np.array([0, 1, 2])
    col_result = np.array([0, 2, 1])
    assert_array_equal(munkres_one[0], row_result)
    assert_array_equal(munkres_one[1], col_result)


def test_min_weight_matching_2():
    """ Test that the correct minimum weight matching is found"""
    # Not saturated case, wide
    munkres_two = linear_sum_assignment(-np.array([[5, 0, 2, 0],
                                                   [1, 3, 4, 0],
                                                   [2, 0, 0, 0]], dtype=float))
    row_result = np.array([0, 1, 2])
    col_result = np.array([0, 2, 1])
    assert_array_equal(munkres_two[0], row_result)
    assert_array_equal(munkres_two[1], col_result)


def test_min_weight_matching_3():
    """ Test that the correct minimum weight matching is found"""
    # Not saturated case, tall
    munkres_three = linear_sum_assignment(-np.array([[5, 0, 2, 0],
                                                     [5, 0, 2, 0],
                                                     [5, 0, 2, 0],
                                                     [1, 3, 4, 0],
                                                     [2, 2, 0, 2]], dtype=float))
    row_result = np.array([0, 1, 3, 4])
    col_result = np.array([0, 2, 1, 3])
    assert_array_equal(munkres_three[0], row_result)
    assert_array_equal(munkres_three[1], col_result)


def test_min_weight_matching_4():
    """ Test that the correct minimum weight matching is found"""
    # Saturated case tall
    munkres_four = linear_sum_assignment(-np.array([[5, 0, 2, 0],
                                                    [5, 0, 2, 0],
                                                    [5, 0, 2, 0],
                                                    [1, 3, 4, 0],
                                                    [2, 2, 0, 2],
                                                    [2, 2, 0, 2]], dtype=float))

    row_result = np.array([0, 3, 4, 5])
    col_result = np.array([0, 2, 3, 1])
    assert_array_equal(munkres_four[0], row_result)
    assert_array_equal(munkres_four[1], col_result)


def test_max_weight_matching_1(matrix_for_tests):
    """ Test that the correct maximum weight matching is found"""
    # Fully saturated case, wide
    munkres_one = linear_sum_assignment(matrix_for_tests, maximize=True)
    row_result = np.array([0, 1, 2])
    col_result = np.array([0, 2, 1])
    assert_array_equal(munkres_one[0], row_result)
    assert_array_equal(munkres_one[1], col_result)


def test_max_weight_matching_2():
    """ Test that the correct maximum weight matching is found"""
    # Not saturated case, wide
    munkres_two = linear_sum_assignment(np.array([[5, 0, 2, 0],
                                                  [1, 3, 4, 0],
                                                  [2, 0, 0, 0]], dtype=float), maximize=True)
    row_result = np.array([0, 1, 2])
    col_result = np.array([0, 2, 1])
    assert_array_equal(munkres_two[0], row_result)
    assert_array_equal(munkres_two[1], col_result)


def test_max_weight_matching_3():
    """ Test that the correct maximum weight matching is found"""
    # Not saturated case, tall
    munkres_three = linear_sum_assignment(np.array([[5, 0, 2, 0],
                                                    [5, 0, 2, 0],
                                                    [5, 0, 2, 0],
                                                    [1, 3, 4, 0],
                                                    [2, 2, 0, 2]], dtype=float), maximize=True)
    row_result = np.array([0, 1, 3, 4])
    col_result = np.array([0, 2, 1, 3])
    assert_array_equal(munkres_three[0], row_result)
    assert_array_equal(munkres_three[1], col_result)


def test_max_weight_matching_4():
    """ Test that the correct maximum weight matching is found"""
    # Saturated case tall
    munkres_four = linear_sum_assignment(np.array([[5, 0, 2, 0],
                                                   [5, 0, 2, 0],
                                                   [5, 0, 2, 0],
                                                   [1, 3, 4, 0],
                                                   [2, 2, 0, 2],
                                                   [2, 2, 0, 2]], dtype=float), maximize=True)

    row_result = np.array([0, 3, 4, 5])
    col_result = np.array([0, 2, 3, 1])
    assert_array_equal(munkres_four[0], row_result)
    assert_array_equal(munkres_four[1], col_result)


def test_transposing(matrix_for_tests):
    """ Test that the putting in the transpose matrix still works"""
    # Saturated case tall
    munkres_four = linear_sum_assignment(np.array([[5, 0, 2, 0],
                                                   [5, 0, 2, 0],
                                                   [5, 0, 2, 0],
                                                   [1, 3, 4, 0],
                                                   [2, 2, 0, 2],
                                                   [2, 2, 0, 2]], dtype=float).transpose(),
                                         maximize=True)

    col_result = np.array([0, 5, 3, 4])
    row_result = np.array([0, 1, 2, 3])
    assert_array_equal(munkres_four[0], row_result)
    assert_array_equal(munkres_four[1], col_result)

    munkres = Munkres(matrix_for_tests)
    assert(munkres.marked.shape == (3, 4))
    munkres = Munkres(matrix_for_tests.transpose())
    assert (munkres.marked.shape == (4, 3))
