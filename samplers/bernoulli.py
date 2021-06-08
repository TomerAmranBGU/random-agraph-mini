import string
from typing import Dict

from samplers.Sampler import Sampler
import numpy as np
import scipy.stats


class Bernoulli(Sampler):
    def __init__(self, p: float, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def sample(self, size: int) -> np.ndarray:
        return scipy.stats.bernoulli.rvs(self.p, size=[size, size])

    def kind_n_dict(self) -> [string,Dict]:
        return  "Bernoulli", {"p": self.p}
