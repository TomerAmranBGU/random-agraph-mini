from longest_path import findLongestPath, Node 
from diag_generator import diag_generator
import numpy as np
n =6

arr = np.ones((n,n))
count =1
# for i, j in diag_generator(n):
#     arr[i,j] = count
#     count +=1

print(arr)
nodes = findLongestPath(arr)

curr =nodes[-1][-1]
while(curr != None):
    arr[curr.i,curr.j] = 0
    curr = curr.pred

print(arr)