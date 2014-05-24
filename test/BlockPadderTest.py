import numpy as np
import unittest
from BlockPadder import blockPad

class BlockPadderTest(unittest.TestCase):

    def testForErrors(self):
        T = np.random.rand(22,35,8)
        T1 = blockPad(T, [16,16,16])    
        
        self.assertTupleEqual((32, 3*16, 16), T1.shape)

        for i0 in range(32):
            for i1 in range(16*3):
                for i2 in range(16):
                    if(i0 < 22 and i1 < 35 and i2 < 8):
                        self.assertEqual(T[i0, i1, i2], T1[i0, i1, i2], 'Values do not match at ' + str(i0) + ' ' + str(i1) + ' ' + str(i2))
                    else:
                        self.assertEqual(0, T1[i0, i1, i2], 'Values do not match at ' + str(i0) + ' ' + str(i1) + ' ' + str(i2))
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
