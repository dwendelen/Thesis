import unittest
import numpy as np
from kr import kr

class KrTest(unittest.TestCase):

    def test_Kr1(self):
        U=[]
        U.append(np.array([[1,2]]))
        U.append(np.array([[1,2],[3,4]]))
        U.append(np.array([[1,2],[3,4],[5,6]]))
        
        R = kr(U)
        S = np.array([[1, 8],
                      [3, 16],
                      [5, 24],
                      [3, 16],
                      [9, 32],
                      [15, 48]])
        
        self.assertTrue(np.array_equal(R, S), "Kronecker test 1 failed, elements do not match")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()