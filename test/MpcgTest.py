import unittest
import numpy as np
from code import *
import numpy.testing as npt
from mpcg import mpcg

def A(x):
    return x*1.1

def M(x):
    return x/2

class MpcgTest(unittest.TestCase):
        
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.b = np.array([1, 2, 3, 8])
        self.x0 = np.array([8, 12, 4, 5])
        
    def test_Mpcg3(self):
        e = np.array([0.9091, 1.8182, 2.7273, 7.2727])
        r = mpcg(A, self.b, M, self.x0, 1e-6, 3)
        
        npt.assert_array_almost_equal(r, e, 4)

    def test_Mpcg6(self):
        e = np.array([0.9091, 1.8182, 2.7273, 7.2727])
        r = mpcg(A, self.b, M, self.x0, 1e-6, 6)
        
        npt.assert_array_almost_equal(r, e, 4)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()