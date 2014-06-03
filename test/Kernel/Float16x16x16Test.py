import numpy as np
import unittest

from Kernel.Float16x16x16.F import F
from Platform.ContextQueue import ContextQueue
from Kernel.Float16x16x16.BufferFactory import BufferFactory

class Float16x16x16Test(unittest.TestCase):

    def testSimple(self):
        
        T = np.zeros((1,2,3));
        T[0, 0, 0] = 111;
        T[0, 1, 0] = 121;
        T[0, 0, 1] = 112;
        T[0, 1, 1] = 122;
        T[0, 0, 2] = 113;
        T[0, 1, 2] = 123;
        
        U=[]
        U.append(np.array([[1,2]]))
        U.append(np.array([[1,2],[3,4]]))
        U.append(np.array([[1,2],[3,4],[5,6]]))
        
        cq = ContextQueue(True)
        cq.init()
        
        b = BufferFactory(cq)
        b.init(T, U)
        
        f = F(cq)
        f.compile()
        f.init(b.T, b.R, b.U, b.I, b.Sum)
        f.run()
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
