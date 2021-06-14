from minimizer import Minimizer
from numpy.core.fromnumeric import reshape
from minimize_longest_path import basicMinimise, priorityMinimise, priorityLogMinimise, positionOnly
import numpy as np
n =5
alpha = 0.5
# weights = np.array([ i + 0.0 for i in range(n*n)]).reshape(n,n)
weights = np.array([ i + 0.0 for i in range(n*n)]).reshape(n,n)
print(weights)

minimzer_types  =['weights',
            'diag',
            'diag_log',
            'double_diag',
            'double_diag_log',
            'position_only']

for type in minimzer_types:
    m = Minimizer(type)
    print(type)
    print(m.minimize(weights ,alpha,put_flag=True))
    print()