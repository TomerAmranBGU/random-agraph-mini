cimport cython
import numpy
cimport numpy
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)   # Deactivate None check.


cpdef void findLongestPath(double[:,:] weights, double[:,:] nodes,int n,diamond=False) :
    # init 
    cdef int pred,i,j
    cdef double current_weight
    if diamond:
        for i in range(n):
            for j in range(n):
                if numpy.abs(i - j) > n / 2 or numpy.abs(i + j - n) > n / 2:
                    weights[i][j] = 0

    # algorithem
    for i in range(n):
        for j in range(n):
            if i == 0 and j > 0:
                nodes[i,j]= weights[i,j]+nodes[i,j-1]

            elif i > 0 and j == 0:
                nodes[i,j]= weights[i,j]+nodes[i - 1,j]

            elif i > 0 and j > 0:
                if nodes[i - 1][j] > nodes[i, j - 1]:
                    nodes[i,j]= weights[i,j]+nodes[i - 1,j]
                else:
                    nodes[i,j]= weights[i,j]+nodes[i,j-1]
