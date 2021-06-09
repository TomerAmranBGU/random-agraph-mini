from typing import List

import numpy as np
from diag_generator import diag_generator

class Node:
    def __init__(self, i,j, weight):
        self.i = i
        self.j = j
        self.weight = weight
        self.pred: Node
        self.longest_path =0

def findLongestPath(weights:np.ndarray) -> List[List[Node]]:
    # init 
    n = weights.shape[0]
    nodes = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            nodes[i][j] = Node(i,j,weights[i,j]) 

    # algorithem
    for i in range(n):
        for j in range(n):
            curr = nodes[i][j]
            pred = None
            if (i == 0 and j>0):
                pred = nodes[i][j-1]

            elif (i>0 and j==0):
                pred = nodes[i-1][j]

            elif (i >0 and j>0):
                if(nodes[i-1][j].longest_path > nodes[i][j-1].longest_path):
                    pred = nodes[i-1][j]
                else:
                    pred = nodes[i][j-1]

            curr.longest_path = curr.weight + (pred.longest_path if pred else 0)
            curr.pred = pred
    return nodes





