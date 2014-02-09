import unittest
import numpy as np
from tens2mat import tens2mat

class CodeTest(unittest.TestCase):
        
    def testSimple(self):
        T = np.array([[[111,112,113,114],[121,122,123,124],[131,132,133,134]],[[211,212,213,214],[221,222,223,224],[231,232,233,234]]])
        a0 = tens2mat(T, 0);
        a1 =  tens2mat(T, 1);
        a2 =  tens2mat(T, 2);
        
        e0 = np.array([[111, 121, 131, 112, 122, 132, 113, 123, 133, 114, 124, 134],
                       [211, 221, 231, 212, 222, 232, 213, 223, 233, 214, 224, 234]])
        
        e1 = np.array([[111, 211, 112, 212, 113, 213, 114, 214],
                       [121, 221, 122, 222, 123, 223, 124, 224],
                       [131, 231, 132, 232, 133, 233, 134, 234]])
        
        e2 = np.array([[111, 211, 121, 221, 131, 231],
                       [112, 212, 122, 222, 132, 232],
                       [113, 213, 123, 223, 133, 233],
                       [114, 214, 124, 224, 134, 234]])
        
        self.assertTrue(np.array_equal(a0, e0))
        self.assertTrue(np.array_equal(a1, e1))
        self.assertTrue(np.array_equal(a2, e2))
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()