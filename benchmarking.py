import _munkres
from scipy.optimize import linear_sum_assignment as lsa
from numpy.random import rand
import time
import cProfile


cost_matrix = rand(500, 500)
cProfile.run('lsa(cost_matrix)', sort='tottime')
cProfile.run('_munkres.linear_sum_assignment(cost_matrix)', sort='tottime')
