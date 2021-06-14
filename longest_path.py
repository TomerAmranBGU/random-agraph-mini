from typing import List, Union

import numpy as np


class Node:
    def __init__(self, i, j, weight):
        self.i = i
        self.j = j
        self.weight = weight
        self.pred: Node
        self.longest_path = 0


def findLongestPath(weights: np.ndarray, nodes: List[List[Union[Node, None]]], diamond=False) -> None:
    # init 
    n = weights.shape[0]
    for i in range(n):
        for j in range(n):
            if diamond and (abs(i - j) > n / 2 or abs(i + j - n) > n / 2):
                nodes[i][j] = Node(i, j, 0)
            else:
                nodes[i][j] = Node(i, j, weights[i, j])

    # algorithem
    for i in range(n):
        for j in range(n):
            curr = nodes[i][j]
            pred = None
            if (i == 0 and j > 0):
                pred = nodes[i][j - 1]

            elif (i > 0 and j == 0):
                pred = nodes[i - 1][j]

            elif (i > 0 and j > 0):
                if (nodes[i - 1][j].longest_path > nodes[i][j - 1].longest_path):
                    pred = nodes[i - 1][j]
                else:
                    pred = nodes[i][j - 1]

            curr.longest_path = curr.weight + (pred.longest_path if pred else 0)
            curr.pred = pred
    return nodes
