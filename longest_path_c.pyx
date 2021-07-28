cimport cython
import numpy as np
cimport numpy as cnp
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)   # Deactivate None check.

cpdef void findLongestPath(float [:,::1] weights, float [:,::1]  nodes,int n,diamond=False) :
    # init 
    cdef int i,j
    if  diamond:
        nodes[0,0]=0
    else:
        nodes[0,0]=weights[0,0]

    # algorithem
    for i in range(n):
        for j in range(n):
            if diamond:
                if abs(i - j) > n / 2 or abs(i + j - n) > n / 2:
                    weights[i][j] = 0
            if i == 0 and j > 0:
                nodes[i,j]= weights[i,j]+nodes[i,j-1]

            elif i > 0 and j == 0:
                nodes[i,j]= weights[i,j]+nodes[i - 1,j]

            elif i > 0 and j > 0:
                if nodes[i - 1][j] > nodes[i, j - 1]:
                    nodes[i,j]= weights[i,j]+nodes[i - 1,j]
                else:
                    nodes[i,j]= weights[i,j]+nodes[i,j-1]
