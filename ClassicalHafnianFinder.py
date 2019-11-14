# Carson Gresens
# AEP 4900
# 10/30/19

import numpy as np
from itertools import permutations
from mpmath import factorial as fac

# Find the hafnian of an nxn adjacency matrix

def FindHaf(M):
    n = len(M)
    prefactor = 1/((fac(n//2))*2**(n//2))
    haf  = 0
    
    if n%2 != 0:
        # Note: the hafnian is defined to be 0 for odd n
        return haf
    
    perms = permutations(np.arange(n))     
    hafsum = 0
    # permutatons returns all the needed permutations
    for p in perms:    
        # Since the edges are pairs of points, choose 
        hafprod = 1
        i = 0
        while i < n:
                row, column = p[i], p[i+1]
                hafprod = hafprod*M[row][column]
                i = i+2
        hafsum = hafsum + hafprod    
    
    return prefactor*hafsum


# Test it
A = [[0,0,0,1,0,0],[0,0,0,1,1,0],[0,0,0,1,1,1,],
     [1,1,1,0,0,0],[0,1,1,0,0,0],[0,0,1,0,0,0]]
B = [[-1,1,1,-1,0,0,1,-1], [1,0,1,0,-1,0,-1,-1],[1,1,-1,1,-1,-1,0,-1],
[-1,0,1,-1,-1,1,-1,0],[0,-1,-1,-1,-1,0,0,-1],[0,0,-1,1,0,0,1,1],
[1,-1,0,-1,0,1,1,0],[-1,-1,-1,0,-1,1,0,1]]

print(FindHaf(A))
print(FindHaf(B))