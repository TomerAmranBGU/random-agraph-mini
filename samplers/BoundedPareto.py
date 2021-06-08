import string
from typing import Dict

from samplers.Sampler import Sampler
import numpy as np
import scipy.stats
from sympy.stats import BoundedPareto, sample, sample_iter


class Boundedpareto(Sampler):
    def __init__(self, h: int, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.l = 1
        self.h = h

    def sample(self, size: int) -> np.ndarray:
        X = BoundedPareto('X', self.alpha, self.l, self.h)
        return next(sample_iter(X, size=(size, size), numsamples=size ** 2))

    def kind_n_dict(self) -> [string, Dict]:
        return "Bounded Pareto", {"alpha": self.alpha, "l": self.l, "h": self.h}
