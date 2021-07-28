import string
from typing import Dict

from samplers.Sampler import Sampler
import numpy as np
import scipy.stats
import scipy.stats


class Boundedpareto(Sampler):
    def __init__(self, h: int, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.l = 1
        self.h = h
        self.h_pow_alpha=np.power(self.h, self.alpha)

    # based on wikipedia https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution
    def inverse_dist(self, x):
        return np.power(-(x * self.h_pow_alpha - x - self.h_pow_alpha) / (
               self.h_pow_alpha), -1 / self.alpha)

    def sample(self, size: int) -> np.ndarray:
        X = scipy.stats.uniform.rvs(size=[size, size])
        return np.ascontiguousarray(self.inverse_dist(X),dtype=np.float32)

    def kind_n_dict(self) -> [string, Dict]:
        return "Bounded Pareto", {"alpha": self.alpha, "l": self.l, "h": self.h}
