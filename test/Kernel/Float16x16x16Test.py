import numpy as np
import unittest
from Kernel.Float16x16x16 import Factory

class Float16x16x16Test(unittest.TestCase):

    def testSimple(self):
        kernel = Factory().create(U, T)
        kernel.run()
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
