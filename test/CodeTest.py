import unittest
import numpy as np
from code import *

class CodeTest(unittest.TestCase):
        

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.initU1()
        self.initT1()
        self.N1 = getNbOfDimensions(self.T1)
        self.R1 = getRank(self.U1)
        self.M1 = getM(self.U1, self.T1)

    def initT1(self):
        T = np.zeros((1,2,3));
        T[0, 0, 0] = 111;
        T[0, 1, 0] = 121;
        T[0, 0, 1] = 112;
        T[0, 1, 1] = 122;
        T[0, 0, 2] = 113;
        T[0, 1, 2] = 123;
        self.T1 = T
        self.T1r = copyListOfArray(T)

    def initU1(self):
        U=[]
        U.append(np.array([[1,2]]))
        U.append(np.array([[1,2],[3,4]]))
        U.append(np.array([[1,2],[3,4],[5,6]]))
        self.U1 = U
        self.U1r = copyListOfArray(U)

    def test_f(self):
        r = f(self.U1, self.M1)
        
        self.testUUnchanged()
        self.assertEqual(23337, r, "F is not correct")

    def test_g(self):
        e = []
        e.append(np.array([[-2736, -5712]]))
        e.append(np.array([[-801, -2160], [-645, -1776]]))
        e.append(np.array([[-408, -1224],[-336, -1020], [-264, -816]]))
    
        UHU = calculateUHU(self.U1, self.N1, self.R1)
    
        r = g(self.U1, UHU, self.N1, self.M1)
        
        self.assertListOfArraysEquals(r, e, "Gradient is wrong")
    
    def test_calculateUHU(self):
        e = np.zeros((self.R1, self.R1, self.N1))
        e[:,:,0] = np.array([[1, 2], [2, 4]])
        e[:,:,1] = np.array([[10, 14], [14, 20]])
        e[:,:,2] = np.array([[35, 44], [44, 56]])
        
        r = calculateUHU(self.U1, self.N1, self.R1)
        self.assertTrue(np.array_equal(e, r), "Elements do not match")
    
    def test_jhjx(self):
        pass
    
    def test_blockJacobi(self):
        pass
    
    def test_serialize(self):
        r = serialize(self.U1)
        
        s =  np.array([1, 2, 1, 3, 2, 4, 1, 3, 5, 2, 4, 6])
        
        self.testUUnchanged()
        self.assertTrue(np.array_equal(r, s), "Serialisation is wrong")
        
    def test_serializeAndDeserialize(self):
        r = serialize(self.U1)
        d = structure(self.U1)

        U2 = deserialize(r, d)
        
        self.testUUnchanged()
        self.assertListOfArraysEquals(self.U1, U2, "Serialisation or deserialisation is wrong")
    
    def testUUnchanged(self):
        self.assertListOfArraysEquals(self.U1, self.U1r, "U changed")
        
    def assertListOfArraysEquals(self, a, b, msg):
        self.assertEqual(len(a), len(b), msg + ": The sizes do not match")
        
        for i in range(len(a)):
            self.assertTrue(np.array_equal(a[i], b[i]), msg + ": The elements do not match")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()