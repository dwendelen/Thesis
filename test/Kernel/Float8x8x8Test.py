import numpy as np
import unittest

from Platform.ContextQueue import ContextQueue
from Kernel.Float8x8x8 import *

from Platform.Platform import NumPyPlatform

class Float4x4x4Test(unittest.TestCase):

    def testF(self):
        
        self.do(10, 10, 8, 3, 0.001)
        self.do(200, 150, 50, 10, 600)
        

    def do(self, I0, I1, I2, R, delta):
        T = np.random.rand(I0, I1, I2).astype(np.float32)
        U0 = np.random.rand(I0, R).astype(np.float32)
        U1 = np.random.rand(I1, R).astype(np.float32)
        U2 = np.random.rand(I2, R).astype(np.float32)
        U = (U0, U1, U2)
        
        fac = Factory()
        fac.init()
        fac.setTU(T, U)
        
        m = fac.createRemapper()
        r = fac.createR()
        
        m.run()
        r.run()
        rr = fac.getF()
        
        npp = NumPyPlatform()
        npp.init()
        npp.setT(T)
        npp.setU(U)
        e = npp.f()
        
        self.assertAlmostEqual(rr, e, delta = delta)
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
