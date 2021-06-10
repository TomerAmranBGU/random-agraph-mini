import string
from typing import Dict

from samplers.Sampler import Sampler
import numpy as np
import scipy.stats


class Geo(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p = 1 - np.exp(-1)

    def sample(self, size: int) -> np.ndarray:
        return scipy.stats.geom.rvs(self.p, size=[size, size])

    def kind_n_dict(self) -> [string,Dict]:
        return  "Geometric", {"p": self.p}
