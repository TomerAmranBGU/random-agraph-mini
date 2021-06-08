import string
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plotHistogram(array: np.ndarray, dist: string, params: Dict, n: int):
    plt.hist(array, 10)
    name = dist + str(params) + " N:" + str(n)
    plt.title(name)
    plt.ylabel(' frequency')
    plt.xlabel(' path cost')
    plt.savefig(name+'.png')
    plt.clf()




