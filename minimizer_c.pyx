from random import choices
from scipy.special import softmax
cimport cython
import numpy
cimport numpy
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)   # Deactivate None check.

cpdef double_diag_log_c(int n):
    cdef int i,j
    cdef double divider1,divider2
    denominators = numpy.zeros((n, n))
    cdef double [:,:] denominators_v = denominators
    for i in range(n):
        for j in range(n):
            divider1 = (i+j+1) if (i+j<n) else (2*(n-1) -i -j+1 )
            divider2 = numpy.abs(i-j)
            denominators_v[i,j] = numpy.log((divider1+divider2)+1)
    return 1/denominators

cpdef percentage_of_paths(int n):
    cdef int  i, j
    denominators=numpy.zeros((n,n))
    cdef double [:,:] denominators_v =denominators
    for i in range(n):
        for j in range(n):
            if (i==0 and j==0) or (i == n-1 and j==n-1) :
                denominators_v[i,j]=1
            elif i==0:
                denominators_v[i,j]=0.5*denominators_v[i,j-1]
            elif j == 0:
                denominators_v[i, j] = 0.5 * denominators_v[i-1, j]
            else:
                denominators_v[i, j] = 0.5 * denominators_v[i - 1, j]+0.5*denominators_v[i,j-1]

    return denominators


cpdef iterate_probalistic(int n_resources,double [:,:] weights,probs, indexes,int non_zeros,n):
    cdef int k,i,j,l
    cdef int n2=n*n
    rng = numpy.random.default_rng()
    for l in range(n_resources) :
        indexes = rng.choice(n2, p=softmax(probs))
        i=indexes/n
        j=indexes % n
        weights[i, j] /= 2
        probs[n*i + j] /= 2

cpdef iterate_deteministic(int n_resources,double [:,:] weights,probs, indexes,int non_zeros,n):
    cdef int k, i, j, l
    cdef int n2 = n * n
    for l in range(n_resources):
        indexes = numpy.argmax(probs)
        i = indexes / n
        j = indexes % n
        weights[i, j] /= 2
        probs[n * i + j] /= 2

cpdef iterate_deteministic_batch(int n_resources,double [:,:] weights,probs, indexes,int non_zeros,n):
    cdef int k, i, j, l
    cdef int n2 = n * n
    while n_resources > 0:
        ind = numpy.argsort(probs)
        k = min(non_zeros, n_resources)
        n_resources -= non_zeros
        for m in range(k):
            l=ind[n2-m-1]
            i = l/n
            j = l%n
            weights[i, j] /= 2
            probs[n * i + j] /= 2

cpdef iterate_probalistic_batch(int n_resources,double [:,:] weights,probs, indexes,int non_zeros,n):
    cdef int k, i, j, l, m
    cdef int n2 = n * n
    rng = numpy.random.default_rng()
    while n_resources > 0:
        k = min(non_zeros, n_resources)
        n_resources -= non_zeros
        tuples = choices(indexes,weights=probs,k=k)
        for m in range(k):
            i = tuples[m][0]
            j = tuples[m][1]
            weights[i, j] /= 2
            probs[n * i + j] /= 2
