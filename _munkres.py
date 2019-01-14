import numpy as np
from functools import partial
import copy


"""
 Algorithm that solves the weighted assignment problem (or linear sum assignment problem) for 
 bipartite graphs whose weighted bi-adjacency matrices are not necessarily square.
 
 Definitions: 
    Many times we are performing operations on the graph whose bi-adjacency matrix is formed by
    putting ones in where zeros are in the assignment matrix and zeros elsewhere. We call this
    graph the **0-induced graph**.
    
    If a row has been assigned a job (i.e. column), we say that it is **saturated**. In the 
    algorithm, this is kept track of via the row_saturated and columns saturated vectors.
    
    Marking is done when we determine a minimum vertex cover of the 0-induced graph. We mark
    all vertices reachable by augmenting paths from free row vertices. From the resulting set
    the column vertices are kept and the complement of the row vertices are kept.
 
 The algorithm consists of six main parts:
    1. A pre-processing of the input matrix for easy access in the main algorithms
    2. Adjust weights so that each row has a zero entry. (Step 1 Wikipedia)
    3. The computation of a maximal matching, done greedily. If the maximal matching is maximum,
       we terminate (Step 2 Wikipedia)
    4. Otherwise, we compute a minimum vertex cover of the bipartite graph defined by zeros in
       the weighted bi-adjacency matrix. (Step 3 in Wikipedia)
    5. Subtract smallest unmarked element. Subtract from each other unmarked element and add to 
       every doubly marked element.  Return to step 2 above. (Step 4 Wikipedia)
    6. Once a result is found, an assignment (i.e. a maximum matching) is found from the current
       known maximal matching by finding augmenting paths.
 
"""


# Class that pre-processes various attributes of the cost matrix
# for ease of access in the main algorithm

def linear_sum_assignment(cost_matrix):
    if cost_matrix.shape[0] <= cost_matrix.shape[1]:
        return Munkres(copy.deepcopy(cost_matrix)).maximum_weight_matching()
    else:
        return Munkres(copy.deepcopy(cost_matrix).transpose()).maximum_weight_matching()


class MunkresMatrix(object):
    """
    Class that pre-processes various attributes of the cost matrix
    for ease of access in the main algorithm
    """

    def __init__(self, matrix):
        self.shape = matrix.shape
        self.marked = np.zeros(self.shape, dtype=bool)
        self.row_saturated = np.zeros(self.shape[0], dtype=bool)
        self.col_saturated = np.zeros(self.shape[1], dtype=bool)
        self.row_marked = np.zeros(self.shape[0], dtype=bool)
        self.col_marked = np.zeros(self.shape[1], dtype=bool)
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

    def _maximal_matching(self):
        """Find a maximal matching greedily"""

        # For each row, find the smallest element in that row and
        # subtract it from each element in its row.
        self.matrix -= self.matrix.min(axis=1)[:, np.newaxis]

        # Iterating through each zero-valued matrix entry, if neither the row or column of the
        # entry has been marked, add entry to the matching and mark the its row and column as
        # being assigned.
        for row, col in zip(*np.where(self.matrix == 0)):
            if not self.row_saturated[row] and not self.col_saturated[col]:
                self.marked[row, col] = self.row_saturated[row] = self.col_saturated[col] = True

    def _remove_covers(self):
        self.row_marked *= 0
        self.col_marked *= 0
        self.row_saturated *= 0
        self.col_saturated *= 0
        self.marked *= 0

    def _min_vertex_cover(self):
        """Find a minimum vertex cover of 0-induced graph"""

        # Start with all unsaturated row vertices
        self.row_marked = self.row_saturated == False

        # We keep trying to find new vertices reachable by augmenting paths.
        while True:
            found = self.col_marked.sum()
            # Saturated column neighbors of rows from previous round
            self.col_marked = np.any(self.matrix[self.row_marked] == 0, axis=0)
            # Mark rows that are matched with columns found above
            self.row_marked[np.any(self.marked[:, self.col_marked], axis=1)] = True
            if self.col_marked.sum() == found:
                return

    def _aug_paths(self, forbidden_col=None):
        """Find an augmenting path if one exists from maximal matching."""
        # Rows checked for augmenting paths
        for row in np.where(self.row_saturated == False)[0]:
            path_row, path_col = self._aug_path(row, forbidden_col=forbidden_col)
            if not path_col:
                continue
            if not len(path_row + path_col) % 2:
                for i in range(len(path_row) - 1):
                    self.marked[path_row[i], path_col[i]] = 1
                    self.marked[path_row[i+1], path_col[i]] = 0
                self.marked[path_row[-1], path_col[-1]] = 1
                self.row_saturated[path_row[0]] = self.col_saturated[path_col[-1]] = True

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
            if self.col_saturated[col]:
                # We find row vertex it is matched with
                row_index = np.argmax(self.marked[:, col])

                # We now try to find augmented path from newly found row vertex
                aug_row, aug_col = self._aug_path(row_index, path_row+[row],
                                                  path_col+[col], forbidden_col)

                # If we could not reach augmented path, continue to next vertex
                if not len(aug_col) or self.col_saturated[aug_col[-1]]:
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
            # Subtract the smallest element in each row from every other entry in same row
            # and compute maximal matching of resulting 0-inducted graph.
            # (Steps 1 and 2 in Wikipedia)
            self._maximal_matching()

            # Find a minimum vertex cover of the 0-induced graph (Step 3 in Wikipedia)
            self._min_vertex_cover()

            # If all rows are saturated, find the maximum matching and stop
            if (self.shape[0] - self.row_marked.sum()) + self.col_marked.sum() == self.shape[0]:
                self._aug_paths()
                break

            # Find minimum value of uncovered edge weights
            minval = np.min(self.matrix[self.row_marked][:, self.col_marked != True])

            # Adjust the matrix weights according to Hungarian algorithm (step 4 Wikipedia)
            self.matrix[self.row_marked] -= minval
            self.matrix[:, self.col_marked] += minval

            # Reset process and run again
            self._remove_covers()
        self.marked *= self.matrix_pattern
        return list(zip(*np.where(self.marked > 0)))
