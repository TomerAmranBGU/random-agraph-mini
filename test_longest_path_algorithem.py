from longest_path import findLongestPath, Node
from diag_generator import diag_generator
import numpy as np

from samplers.bernoulli import Bernoulli

n = 6
sampler = Bernoulli(0.5)
arr = sampler.sample(n)
count = 1
# for i, j in diag_generator(n):
#     arr[i,j] = count
#     count +=1

print(arr)
nodes = findLongestPath(arr)

curr = nodes[-1][-1]
while (curr != None):
    arr[curr.i, curr.j] = -1
    curr = curr.pred

print(arr,nodes[-1][-1].longest_path)
