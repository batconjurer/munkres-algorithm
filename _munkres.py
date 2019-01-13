import numpy as np
from functools import partial
import copy


"""===================Sparse Matching Class==================="""


# A class optimized for determining sizes of maximum matchings and solving the assignment problem.
class MunkresMatrix(object):

    def __init__(self, matrix):
        self.shape = matrix.shape
        self.marked = np.zeros(self.shape, dtype=bool)
        self.row_unmarked = np.ones(self.shape[0], dtype=bool)
        self.col_unmarked = np.zeros(self.shape[1], dtype=bool)
        self.non_zero_pairs = list(zip(*np.where(matrix != 0)))
        self.matrix = matrix
        self._columns = {}
        self._rows = {}
        self._make_rows_and_columns()
        self.matrix_pattern = (self.matrix != 0).astype(int)

    def _to_sort_column(self, row, col):
        return -self.matrix[row, col]

    def _to_sort_row(self, col, row):
        return -self.matrix[row, col]

    def _make_rows_and_columns(self):
        for pair in self.non_zero_pairs:
            if self._rows.get(pair[0]) is not None:
                self._rows[pair[0]].append(pair[1])
            else:
                self._rows[pair[0]] = [pair[1]]
            if self._columns.get(pair[1]) is not None:
                self._columns[pair[1]].append(pair[0])
            else:
                self._columns[pair[1]] = [pair[0]]
        for key in self._rows:
            self._rows[key].sort(key=partial(self._to_sort_row, row=key))
            self._rows[key] = np.array(self._rows[key], dtype=int)
        for key in self._columns:
            self._columns[key].sort(key=partial(self._to_sort_column, col=key))
            self._columns[key] = np.array(self._columns[key], dtype=int)


class Munkres(MunkresMatrix):
    """Class for finding maximum weight matchings and minimum vertex covers in bipartite graph"""

    def __init__(self, matrix):
        matrix = -matrix
        MunkresMatrix.__init__(self, matrix)
        self.match_size = min(self.matrix.shape[0], self.matrix.shape[1])

    def _maximal_matching(self):
        """Find a maximal matching greedily"""

        # For each row, find the smallest element in that row and
        # subtract it from each element in its row.
        self.matrix -= self.matrix.min(axis=1)[:, np.newaxis]

        # Iterating through each matrix entry, if neither the row or column of the entry
        # has been marked, add entry to the matching and mark the its row and column as
        # being assigned.
        for row, col in zip(*np.where(self.matrix == 0)):
            if self.row_unmarked[row] and self.col_unmarked[col]:
                self.marked[row, col] = True
                self.col_unmarked[col] = self.row_unmarked[row] = False

    def _remove_covers(self):
        self.row_unmarked = np.ones(self.shape[0], dtype=bool)
        self.col_unmarked = np.ones(self.shape[1], dtype=bool)
        self.marked[:, :] = 0

    def min_vertex_cover(self):
        """Find a minimum vertex cover"""
        # Rows with no assignments (free vertices)
        # Initially no columns are found
        col_found = np.zeros(self.shape[1], dtype=bool)

        # We keep trying to find new vertices reachable by augmenting paths.
        while True:
            found = col_found.sum()
            # Saturated column neighbors of rows from previous round
            col_found = np.any(self.matrix[self.row_unmarked] == 0, axis=0)
            # Mark rows that are matched with columns found above
            self.row_unmarked[np.any(self.marked[:, col_found], axis=1)] = True
            if col_found.sum() == found:
                break

        return col_found

    def _aug_paths(self, forbidden_col=None):
        """Find an augmenting path if one exists from maximal matching."""
        # Rows checked for augmenting paths
        for row in np.where(self.row_unmarked)[0]:
            path_row, path_col = self._aug_path(row, forbidden_col=forbidden_col)
            if not path_col:
                continue
            if not len(path_row + path_col) % 2:
                for i in range(len(path_row) - 1):
                    self.marked[path_row[i], path_col[i]] = 1
                    self.marked[path_row[i+1], path_col[i]] = 0
                self.marked[path_row[-1], path_col[-1]] = 1
                self.row_unmarked[path_row[0]] = self.col_unmarked[path_col[-1]] = False

    def _aug_path(self, row, path_row=None, path_col=None, forbidden_col=None):
        if path_row is None:
            path_row = []
        if path_col is None:
            path_col = []

        # We now check every column to see if we can extend augmented path with column vertex
        if self._rows.get(row) is None:
            return [], []

        for col in self._rows[row]:
            # We do not check vertices already on path
            if col in path_col:
                continue
            if col == forbidden_col:
                continue
            # If vertex is marked, it we check to see if it can extend path
            if not self.col_unmarked[col]:
                # We find row vertex it is matched with
                row_index = np.argmax(self.marked[:, col])

                # We now try to find augmented path from newly found row vertex
                aug_row, aug_col = self._aug_path(row_index, path_row+[row],
                                                  path_col+[col], forbidden_col)

                # If we could not reach augmented path, continue to next vertex
                if not len(aug_col) or not self.col_unmarked[aug_col[-1]]:
                    continue
                else:
                    # If we succeeded in finding augmented path, return it
                    return aug_row, aug_col
            else:
                # We have found the end of an augmented path. We add column vertex.
                return path_row+[row], path_col+[col]

        # If no extension of augmented paths could be found
        return path_row[:-1], path_col[:-1]

    def maximum_weight_matching(self):
        """Main algorithm. Runs the Hungarian Algorithm."""
        while True:

            # Find a minimum vertex cover
            col_found = self.min_vertex_cover()

            # If all rows are saturated, find the maximum matching and stop
            if self.row_unmarked.sum() + col_found.sum() == self.match_size:
                self._aug_paths()
                break
            # Find minimum value of uncovered edge weights
            minval = np.min(self.matrix[self.row_unmarked][:, col_found != True])

            # Adjust the matrix weights according to Hungarian algorithm
            self.matrix[self.row_unmarked] -= minval
            self.matrix[:, col_found] += minval

            # Reset process and run again
            self._remove_covers()
            self._maximal_matching()
        self.marked *= self.matrix_pattern
        return list(zip(*np.where(self.marked > 0)))
