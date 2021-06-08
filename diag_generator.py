import numpy as np
def diag_generator(n):
    for diag in range(n):
        i = diag
        j= 0
        while (i>-1):
            yield i,j
            i -=1
            j +=1
    for diag in range(1,n):
        j = diag
        i = n-1
        while(j<n):
            yield i,j
            j+=1
            i-=1


            