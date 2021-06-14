import numpy as np
from typing import List, Set, Dict, Tuple, Optional

from minimizer import Minimizer

n =10
alpha = 0.5
weights = np.array([ i + 0.0 for i in range(n*n)]).reshape(n,n)
print(weights)

minimizer = Minimizer()
print(minimizer.minimize(weights,alpha, ,put_flag=True))