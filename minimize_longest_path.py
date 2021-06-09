import numpy as np
from typing import List, Set, Dict, Tuple, Optional

class NodePriority:
    def __init__(self, i, j, priority):
        self.i =i
        self.j =j
        self.priority = priority

def sortByWeight(weights)->List[NodePriority]:
    n = weights.shape[0]
    queue = []
    for i in range(n):
        for j in range(n):
            weight = weights[i,j]
            priority = weight
            queue.append( NodePriority(i,j,priority))
    queue.sort(key=lambda x : x.priority, reverse=True)
    return queue

def sortByWeightAndNumOfPaths(weights)->List[NodePriority]:
    n = weights.shape[0]
    queue = []
    for i in range(n):
        for j in range(n):
            weight = weights[i,j]
            divider = (i+j+1) if (i+j<n) else (2*(n-1) -i -j+1 )
            priority = weight / divider
            queue.append( NodePriority(i,j,priority))
    queue.sort(key=lambda x : x.priority,reverse=True)
    return queue
def sortByWeightAndLogNumOfPaths(weights)->List[NodePriority]:
    n = weights.shape[0]
    queue = []
    for i in range(n):
        for j in range(n):
            weight = weights[i,j]
            divider = (i+j+1) if (i+j<n) else (2*(n-1) -i -j+1 )
            divider = np.log(1+divider)
            priority = weight / divider
            queue.append( NodePriority(i,j,priority))
    queue.sort(key=lambda x : x.priority,reverse=True)
    return queue

# sort only by position except nodes that their wight is zero
def sortByPositionOnly(weights)->List[NodePriority]:
    n = weights.shape[0]
    queue = []
    for i in range(n):
        for j in range(n):
            weight = 1 if (weights[i,j]>0) else 0
            divider = (i+j+1) if (i+j<n) else (2*(n-1) -i -j+1 )
            priority = weight / divider
            queue.append( NodePriority(i,j,priority))
    queue.sort(key=lambda x : x.priority,reverse=True)
    return queue
flag = 0
def minimize(weights, alpha, sorter,put_flag=False):
    weights = np.ndarray.copy(weights)
    n = weights.shape[0]
    n_resources = int(n*n*alpha)
    queue = sorter(weights)
    for i in range(n_resources):
        node = queue[i]
        weight =  weights[node.i][node.j] / 2 
        if put_flag:
            weight = flag
        weights[node.i][node.j] = weight
    return weights


def basicMinimise(weights, alpha, put_flag=False):
    return minimize(weights, alpha, sortByWeight,put_flag=put_flag)

def priorityMinimise(weights, alpha,put_flag=False):
    return minimize(weights, alpha, sortByWeightAndNumOfPaths,put_flag=put_flag)
   
def priorityLogMinimise(weights, alpha,put_flag=False):
    return minimize(weights, alpha, sortByWeightAndLogNumOfPaths,put_flag=put_flag)

def positionOnly(weights, alpha,put_flag=False):
    return minimize(weights, alpha, sortByPositionOnly,put_flag=put_flag)
