import unittest
import numpy as np
from code import *

class CodeTest(unittest.TestCase):
        

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.initU1()
        self.initT1()

    def initT1(self):
        T = np.zeros((1,2,3));
        T[0, 0, 0] = 111;
        T[0, 1, 0] = 121;
        T[0, 0, 1] = 112;
        T[0, 1, 1] = 122;
        T[0, 0, 2] = 113;
        T[0, 1, 2] = 123;
        self.T1 = T

    def initU1(self):
        U=[]
        U.append(np.array([[1,2]]))
        U.append(np.array([[1,2],[3,4]]))
        U.append(np.array([[1,2],[3,4],[5,6]]))
        self.U1 = U

    def test_f(self):
        r = f(self.U1, getM(self.U1, self.T1))
        self.assertEqual(23337, r, "F is not correct")

    def test_g(self):
        pass
    
    def test_jhjx(self):
        pass
    
    def test_blockJacobi(self):
        pass
    
    def test_serialize(self):
        r = serialize(self.U1)
        
        s =  np.array([1, 2, 1, 3, 2, 4, 1, 3, 5, 2, 4, 6])
        
        self.assertTrue(np.array_equal(r, s), "Serialisation is wrong")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()