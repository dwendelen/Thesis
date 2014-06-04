import numpy as np
import unittest

from Kernel.Float16x16x16.F import F
from Platform.ContextQueue import ContextQueue
from Kernel.Float16x16x16.BufferFactory import BufferFactory
from Kernel.Float16x16x16.Factory import Factory

from Platform.Platform import NumPyPlatform

class Float16x16x16Test(unittest.TestCase):

    def testF(self):
        
        T = np.random.rand((200, 150, 50))
        U0 = np.random.rand((200, 10))
        U1 = np.random.rand((150, 10))
        U2 = np.random.rand((50, 10))
        U = (U0, U1, U2)
        
        fac = Factory()
        fac.init()
        fac.setTU(T, U)
        f = fac.createF()
        f.run()
        
        r = fac.getF()
        
        npp = NumPyPlatform()
        npp.init()
        npp.setT(T)
        npp.setU(U)
        r = npp.f()
        
        self.assertEqual(r, e)
        
    def testR(self):
        
        T = np.random.rand(200, 150, 50).astype(np.float32)
        U0 = np.random.rand(200, 10).astype(np.float32)
        U1 = np.random.rand(150, 10).astype(np.float32)
        U2 = np.random.rand(50, 10).astype(np.float32)
        U = (U0, U1, U2)
        
        fac = Factory()
        fac.init()
        fac.setTU(T, U)
        
        m = fac.createRemapper()
        r = fac.createF()
        
        m.run()
        r.run()
        
        r = fac.getF()
        
        npp = NumPyPlatform()
        npp.init()
        npp.setT(T)
        npp.setU(U)
        r = npp.f()
        
        self.assertEqual(r, e)
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
