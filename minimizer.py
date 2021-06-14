import numpy as np
from typing import List, Set, Dict, Tuple, Optional

class NodePriority:
    def __init__(self, i, j, priority):
        self.i =i
        self.j =j
        self.priority = priority


class Minimizer:
    def __init__(self, priority_type,flag =0):
        self.priority_type_map = {
            'no_minimizer': None,
            'weights': self.weights,
            'diag': self.diag,
            'diag_log': self.diag_log,
            'double_diag': self.double_diag,
            'double_diag_log': self.double_diag_log,
            'position_only': self.position_only
        }
        self.prioriter = self.priority_type_map[priority_type]
        self.priority_type = priority_type
        self.flag = flag
    
    
    def minimize(self,weights, alpha ,put_flag=False):
        if self.priority_type == 'no_minimizer':
            return weights
        prioriter = self.prioriter
        weights = np.ndarray.copy(weights)
        n = weights.shape[0]
        n_resources = int(n*n*alpha)
        queue = self.sort(weights, prioriter,n)
        for i in range(n_resources):
            node = queue[i]
            weight =  weights[node.i][node.j] / 2 
            if put_flag:
                weight = self.flag
            weights[node.i][node.j] = weight
        return weights
    
    
    def sort(self,weights, prioriter, n)->List[NodePriority]:
        queue = []
        for i in range(n):
            for j in range(n):
                weight = weights[i,j]
                priority = prioriter(i,j,weight,n)
                queue.append( NodePriority(i,j,priority))
        queue.sort(key=lambda x : x.priority, reverse=True)
        return queue



    def weights(self,i,j,weight,n):
        return weight
    
    def diag(self,i,j,weight,n):
        divider = (i+j+1) if (i+j<n) else (2*(n-1) -i -j+1 )
        priority = weight / divider
        return priority
    
    def diag_log(self,i,j,weight,n):
        divider = (i+j+1) if (i+j<n) else (2*(n-1) -i -j+1 )
        divider = np.log(1+divider)
        priority = weight / divider
        return priority
    def double_diag(self,i,j,weight,n):
        divider1 = (i+j+1) if (i+j<n) else (2*(n-1) -i -j+1 )
        divider2 = np.abs(i-j)
        priority = weight / (divider1+divider2)
        return priority

    def double_diag_log(self,i,j,weight,n):
        divider1 = (i+j+1) if (i+j<n) else (2*(n-1) -i -j+1 )
        divider2 = np.abs(i-j)
        priority = weight / np.log((divider1+divider2)+1)
        return priority

    def position_only(self,i,j,weight,n):
        weight = 1 if (weight>0) else 0
        divider1 = (i+j+1) if (i+j<n) else (2*(n-1) -i -j+1 )
        divider2 = np.abs(i-j)
        priority = weight / (divider1+divider2)
        return priority   
