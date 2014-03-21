import unittest
import numpy as np
from code import *
import numpy.testing as npt

class IndexAlgoritmeTest(unittest.TestCase):
    
    def test1(self):
        size = (5,8,6, 3)
        l = (1, 5, 5*8, 5*8*6, 5*8*6*3)
        
        T = np.zeros(size)
        
        for i0 in range(size[0]):
            for i1 in range(size[1]):
                for i2 in range(size[2]):
                    for i3 in range(size[3]):
                        T[i0, i1, i2, i3] = ((((i0+1) * 10) + (i1+1)) *10 + (i2+1)) * 10 + (i3+1)
        
        vec = T.flatten('F')
        
        for n in range(4):
            F = tens2mat(T, n)
            for i in range(F.shape[0]):
                for j in range(F.shape[1]):
                    ind = self.alg(i, j, l[n], l[n+1])
                    self.assertEqual(F[i, j], vec[ind], 
                                     "F: " + str(F[i,j]) + " vec: " + str(vec[ind]) + 
                                     " ind: " + str(ind) + " i: " + str(i) + " j: " + str(j))
        
    def alg(self, slice, i, li, lip1):
        return slice*li + lip1*((int(i)/int(li))) + i % li;

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()