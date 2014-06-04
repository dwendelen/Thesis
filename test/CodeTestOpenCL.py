import unittest
import numpy as np
from code import *
import numpy.testing as npt
import scipy.io
from Platform.OpenCLPlatform import *
from Platform.ContextQueue import ContextQueue
from Kernel.Float16x16x16.F import F
from Kernel.Float16x16x16.BufferFactory import BufferFactory
from Kernel.NumpySum import NumpySum

class CodeTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.initU1()
        self.initT1()
        self.initB2()
        self.N1 = getNbOfDimensions(self.T1)
        self.R1 = getRank(self.U1)
        self.M1 = getM(self.T1)
        self.UHU1 = calculateUHU(self.U1, self.N1, self.R1)
        self.offset1 = calculateOffset(getDimensions(self.T1), self.R1)
        
        self.N2 = getNbOfDimensions(self.T2)
        self.R2 = getRank(self.U2)
        self.M2 = getM(self.T2)
        self.UHU2 = calculateUHU(self.U2, self.N2, self.R2)
        self.offset2 = calculateOffset(getDimensions(self.T2), self.R2)
        
        self.OpenCLPlatform = OpenCLPlatform()
        self.OpenCLPlatform.init()
        self.OpenCLPlatform.setT(self.T1)

    def initT1(self):
        T = np.zeros((1,2,3), dtype = np.float32);
        T[0, 0, 0] = 111;
        T[0, 1, 0] = 121;
        T[0, 0, 1] = 112;
        T[0, 1, 1] = 122;
        T[0, 0, 2] = 113;
        T[0, 1, 2] = 123;
        self.T1 = T
        self.T2 = T
        self.T1r = copyListOfArray(T)

    def initU1(self):
        U=[]
        U.append(np.array([[1,2]], dtype = np.float32))
        U.append(np.array([[1,2],[3,4]], dtype = np.float32))
        U.append(np.array([[1,2],[3,4],[5,6]], dtype = np.float32))
        
        self.U1 = U
        self.U1r = copyListOfArray(U)
        
        U2=[]
        U2.append(np.array([[101,102]], dtype = np.float32))
        U2.append(np.array([[201,202],[203,204]], dtype = np.float32))
        U2.append(np.array([[301,302],[303,304],[305,306]], dtype = np.float32))
        
        self.U2 = U2

    def initB2(self):
        self.b2 = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112], dtype = np.float32)

    def test_g(self):
        e = []
        e.append(np.array([[-2736, -5712]]))
        e.append(np.array([[-801, -2160], [-645, -1776]]))
        e.append(np.array([[-408, -1224],[-336, -1020], [-264, -816]]))
    
        UHU = calculateUHU(self.U1, self.N1, self.R1)
        
        self.OpenCLPlatform.setU(self.U1)
        self.OpenCLPlatform.setUHU(UHU)
    
        r = self.OpenCLPlatform.g()
        
        self.assertListOfArraysEquals(r, e, "Gradient is wrong")

    
    def test_f_16x16x16(self):
        exp = 23337
        
        cq = ContextQueue()
        cq.init()
        
        b = BufferFactory(cq)
        b.init(self.T1, self.U1)
        
        f = F(cq)
        f.compile()
        f.init(b.T, b.R, b.U, b.I, b.Sum)
        f.run()
        
        nps = NumpySum(cq.queue)
        nps.init(b.Sum)
        
        array = np.zeros((16,2), dtype = np.float32)
        cl.enqueue_copy(cq.queue, array, b.U[0])
        print array
        cl.enqueue_copy(cq.queue, array, b.U[1])
        print array
        cl.enqueue_copy(cq.queue, array, b.U[2])
        print array
        
        array = np.zeros((16,16,16), dtype = np.float32)
        cl.enqueue_copy(cq.queue, array, b.T)
        print array
        
        r = nps.getF()
        
        self.testUUnchanged()
        self.assertEqual(exp, r, "F is not correct")
    
    def testUUnchanged(self):
        self.assertListOfArraysEquals(self.U1, self.U1r, "U changed")
        
    def assertListOfArraysEquals(self, a, b, msg):
        self.assertEqual(len(a), len(b), msg + ": The sizes do not match")
        
        for i in range(len(a)):
            npt.assert_array_equal(a[i], b[i], msg + ": The elements do not match")
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
