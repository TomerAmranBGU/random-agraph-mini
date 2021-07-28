from random import choices
from scipy.special import softmax
cimport cython
import numpy as np
cimport numpy as cnp
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)   # Deactivate None check.



cdef  int min2(int n1, int n2):
    if n1>n2:
        return n2
    return n1

cpdef cnp.ndarray[cnp.float32_t, ndim=2] double_diag_log_c(int n):
    cdef int i,j
    cdef float divider1,divider2
    cdef cnp.ndarray[cnp.float32_t, ndim=2] denominators = np.zeros((n, n),dtype= np.float32)
    cdef float[:,::1] denominators_v = denominators
    for i in range(n):
        for j in range(n):
            divider1 = (i+j+1) if (i+j<n) else (2*(n-1) -i -j+1 )
            divider2 = abs(i-j)
            denominators[i,j] = np.log((divider1+divider2)+1)
    return 1/denominators



cpdef void iterate_deteministic_batch(int n_resources,float [:,::1]  weights,float [::1]  probs, int non_zeros,int n):
    cdef int k, i, j, l
    cdef int n2 = n * n
    while n_resources > 0:
        ind = np.argsort(probs)
        k = min2(non_zeros, n_resources)
        n_resources -= k
        for m in range(k):
            l=ind[n2-m-1]
            i = l/n
            j = l%n
            weights[i, j] /= 2
            if n_resources > 0:
                probs[n * i + j] /= 2


