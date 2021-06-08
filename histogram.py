import string
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plotHistogram(array: np.ndarray, dist: string, params: Dict, n: int):
    plt.hist(array, 20)
    plt.title(dist + str(params) + " N:" + str(n))
    plt.ylabel(' y')
    plt.xlabel(' x')
    plt.show()


plotHistogram(np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27]), "exp", {"alpha": 1}, 10)
