import numpy as np
import numpy.linalg as la
from kr import kr
from code import *
from math import *

'''
PLATFORM
--------

INTERFACE
f(U)
init()
setU()
'''
class Platform:    
    def setT(self, T):
        self.T = T;
        self.N = getNbOfDimensions(T)
        self.I = getDimensions(T)
        
    def setU(self, U):
        self.U = U
        self.R = getRank(U)
    
    def setUHU(self, UHU):
        self.UHU = UHU
        
class NumPyPlatform (Platform):
    
    def init(self):
        return
    
    def setT(self, T):
        Platform.setT(self, T)
        self.M = getM(T)
    
    def f(self):
        D = self.M[0] - self.U[0].dot(kr(self.U[:0:-1]).T)
        fval = 0.5 * np.sum(D*D)
        return fval
   
    def g(self):
        r = np.array(range(self.N))
        grad = [];
        
        for n in r:
            allButN = np.hstack((r[:n], r[n+1:self.N]))
            
            UallButNRev = []
            for i in reversed(allButN):
                UallButNRev .append(self.U[i])
            
            G1 = self.U[n].dot(np.prod(self.UHU[:,:,allButN], axis = 2))
            G2 = self.M[n].dot(kr(UallButNRev ))
            grad.append(G1-G2)
            
        return grad
