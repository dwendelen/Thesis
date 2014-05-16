import unittest
import numpy as np
from code import *
import numpy.testing as npt
import scipy.io
from Platform import NumPyPlatform#, OpenCLPlatform

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

    def initT1(self):
        T = np.zeros((1,2,3));
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
        U.append(np.array([[1,2]]))
        U.append(np.array([[1,2],[3,4]]))
        U.append(np.array([[1,2],[3,4],[5,6]]))
        
        self.U1 = U
        self.U1r = copyListOfArray(U)
        
        U2=[]
        U2.append(np.array([[101,102]]))
        U2.append(np.array([[201,202],[203,204]]))
        U2.append(np.array([[301,302],[303,304],[305,306]]))
        
        self.U2 = U2

    def initB2(self):
        self.b2 = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112])

    def test_g(self):
        npPlatform = NumPyPlatform(self.T1)
        npPlatform.init()
        self.g_impl(npPlatform)
        
        '''clPlatform = OpenCLPlatform(self.T1)
        clPlatform.init()
        self.g_impl(clPlatform)'''
        
    def g_impl(self, platform):
        e = []
        e.append(np.array([[-2736, -5712]]))
        e.append(np.array([[-801, -2160], [-645, -1776]]))
        e.append(np.array([[-408, -1224],[-336, -1020], [-264, -816]]))
    
        UHU = calculateUHU(self.U1, self.N1, self.R1)
        platform.setU(self.U1)
        platform.setUHU(UHU)
    
        r = platform.g()
        
        self.assertListOfArraysEquals(r, e, "Gradient is wrong")
    
    def test_f(self):
        npPlatform = NumPyPlatform(self.T1)
        npPlatform.init()
        self.f_impl(npPlatform)
        
        '''clPlatform = OpenCLPlatform(self.T1)
        clPlatform.init()
        self.f_impl(clPlatform)'''
    
    def f_impl(self, platform):
        exp = 23337
        platform.setU(self.U1)
        r = platform.f()
        self.testUUnchanged()
        self.assertEqual(exp, r, "F is not correct")
    
    def test_offset(self):
        r = np.array([0, 2, 6, 12])
        s = calculateOffset(getDimensions(self.T1), self.R1)
    
        self.assertTrue(np.array_equal(r, s))
    
    '''def test_g(self):
        e = []
        e.append(np.array([[-2736, -5712]]))
        e.append(np.array([[-801, -2160], [-645, -1776]]))
        e.append(np.array([[-408, -1224],[-336, -1020], [-264, -816]]))
    
        UHU = calculateUHU(self.U1, self.N1, self.R1)
    
        r = g(self.U1, UHU, self.N1, self.M1)
        
        self.assertListOfArraysEquals(r, e, "Gradient is wrong")'''
    
    def test_calculateUHU(self):
        e = np.zeros((self.R1, self.R1, self.N1))
        e[:,:,0] = np.array([[1, 2], [2, 4]])
        e[:,:,1] = np.array([[10, 14], [14, 20]])
        e[:,:,2] = np.array([[35, 44], [44, 56]])
        
        r = calculateUHU(self.U1, self.N1, self.R1)
        self.assertTrue(np.array_equal(e, r), "Elements do not match")
    
    def test_jhjx(self):
        x = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144])
        r = np.array([35539, 64740, 5047, 10164,
                      12876, 25932, 1835, 3053,
                      4909, 5240, 8720, 14020])
        s = JHJx(self.U1, self.UHU1, self.N1, self.R1, self.offset1, getDimensions(self.T1), x)
        npt.assert_array_equal(r, s)
    
    def test_blockJacobi(self):
        
        r = np.array([-0.0080, 0.0079, -0.7119, -0.6971,
                      0.7026, 0.6880, -2.7976, -2.7680,
                      -2.7383, 2.7565, 2.7273, 2.6981])
        
        s = M_blockJacobi(self.b2, self.N2, self.UHU2, self.offset2, getDimensions(self.T2), self.R2)
        
        npt.assert_array_almost_equal(r, s, 4)
    
    
    
    def test_serialize(self):
        r = serialize(self.U1)
        
        s =  np.array([1, 2, 1, 3, 2, 4, 1, 3, 5, 2, 4, 6])
        
        self.testUUnchanged()
        npt.assert_array_equal(r, s, "Serialisation is wrong")
        
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
            npt.assert_array_equal(a[i], b[i], msg + ": The elements do not match")


    def testFull(self):
        
        '''
        Generate test cases with:
        size_tens = [4 6 8 2];
        R = 4;
        
        for i = 1:5
            U = cpd_rnd(size_tens, R);
            T = cpdgen(U);
            
            options.Algorithm = @nls_gndl;
            options.M = 'block-Jacobi';
            options.LargeScale = true;
            
            S = cpd_nls(T,U,options);
            
            save(strcat('test',int2str(i)), 'U', 'T', 'S', 'size_tens', 'R');
        end
        '''
        
        for i in range(1, 6):
            mat = scipy.io.loadmat('test' + str(i) + '.mat');
            size_tens = mat['size_tens']
            R = mat['R']
            S = self.accessList(mat, 'S')
            U = self.accessList(mat, 'U')
            T = mat['T']
            S1 = cpd_nls(T, U)
            self.assertListOfArraysEquals(S, S1, 'Test ' + str(i) + ' Failed.')
            
    def accessList(self, mat, name):
        return mat[name][0].tolist()
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()