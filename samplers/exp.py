import string
from typing import Dict

from samplers.Sampler import Sampler
import numpy as np
import scipy.stats


class Expon(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = 1

    def sample(self, size: int) -> np.ndarray:
        return scipy.stats.expon.rvs(self.alpha, size=[size, size]).astype(np.double)

    def kind_n_dict(self) -> [string, Dict]:
        return "Exponential", {"alpha": 1}
