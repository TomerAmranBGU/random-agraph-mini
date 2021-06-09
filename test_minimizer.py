from numpy.core.fromnumeric import reshape
from minimize_longest_path import basicMinimise, priorityMinimise, priorityLogMinimise
import numpy as np
n =10
alpha = 0.5
weights = np.array([ i + 0.0 for i in range(n*n)]).reshape(n,n)
print(weights)

print(basicMinimise(weights, alpha))
print(priorityMinimise(weights, alpha))
print(priorityLogMinimise(weights, alpha))