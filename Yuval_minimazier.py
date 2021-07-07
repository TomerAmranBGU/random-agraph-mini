import itertools
import random
import numpy as np

from minimizer_c import percentage_of_paths, iterate_probalistic, double_diag_log_c, iterate_deteministic, \
    iterate_probalistic_batch,iterate_deteministic_batch

priority_type_map = {
    'percentage_of_paths': percentage_of_paths,
    'double_diag_log': double_diag_log_c
}


class YMinimizer:
    def __init__(self, priority_type, n: int):
        self.priority_type = priority_type
        self.denominators = double_diag_log_c(n)
        self.indexes = list(itertools.product(range(n), range(n)))
        self.n = n

    def minimize(self, weights, alpha):
        if alpha == 0:
            return
        n_resources = int(self.n * self.n * alpha)
        probs = weights * self.denominators
        iterate_deteministic_batch(n_resources, weights, probs.reshape(-1), self.indexes, np.count_nonzero(weights), self.n)
