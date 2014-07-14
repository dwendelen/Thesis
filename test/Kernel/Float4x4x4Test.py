import numpy as np
import unittest

from Platform.ContextQueue import ContextQueue
from Kernel.Float4x4x4.BufferFactory import BufferFactory
from Kernel.Float4x4x4.Factory import Factory

from Platform.Platform import NumPyPlatform

class Float4x4x4Test(unittest.TestCase):

    def testF(self):
        
        T = np.random.rand(200, 150, 50).astype(np.float32)
        U0 = np.random.rand(200, 10).astype(np.float32)
        U1 = np.random.rand(150, 10).astype(np.float32)
        U2 = np.random.rand(50, 10).astype(np.float32)
        U = (U0, U1, U2)
        
        fac = Factory()
        fac.init()
        fac.setTU(T, U)
        
        #f = fac.createF()
        m = fac.createRemapper()
        r = fac.createR()
        
        #f.run()
        #rf = fac.getF()
        
        m.run()
        r.run()
        rr = fac.getF()
        
        npp = NumPyPlatform()
        npp.init()
        npp.setT(T)
        npp.setU(U)
        e = npp.f()
        
        print rr
        print e
        
        #self.assertEqual(rf, rr)
        self.assertAlmostEqual(rr, e, delta = 600)
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
