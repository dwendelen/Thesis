import pyopencl as cl
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
    def __init__(self, T):
        self.T = T;
        self.N = getNbOfDimensions(T)
        self.I = getDimensions(T)
        
    def setU(self, U):
        self.U = U
        self.R = getRank(U)
    
    def setUHU(self, UHU):
        self.UHU = UHU
        
class NumPyPlatform (Platform):

    def __init__(self, T):
        Platform.__init__(self, T)
        self.M = getM(T)
    
    def init(self):
        return
    
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
    
class OpenCLPlatform (Platform):
    def __init__(self, T):
        Platform.__init__(self, T)
        
    def init(self):
        devices = cl.get_platforms()[0].get_devices(cl.device_type.GPU)
        context = cl.Context([devices[0]])
        queue = cl.CommandQueue(context)
        
        file = open('../opencl/16x16x16float.cl', 'r')
        
        prg = cl.Program(context, file.read()).build()
            
        self.prg = prg
        self.queue = queue
        self.context = context
    
    def f(self):
        U0 = np.array(self.U[0], order='F', dtype = np.float32)
        U1 = np.array(self.U[1], order='F', dtype = np.float32)
        U2 = np.array(self.U[2], order='F', dtype = np.float32)
        
        T = np.array(self.T, order='F', dtype=np.float32)
        print 'Create Buffers'
        mf = cl.mem_flags
        T_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=T)
        U0_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=U0)
        U1_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=U1)
        U2_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=U2)
        l_buf = cl.Buffer(self.context, mf.WRITE_ONLY, size = 64)
        print 'Buffers created'
        
        print 'Launch'
        
        g = (ceil(T.shape[0]/4.0), ceil(T.shape[1]/4.0), ceil(T.shape[2]/4.0))
        
        self.prg.float16x16x16(self.queue, g, (4,4,4), T_buf, U0_buf, U1_buf, U2_buf, l_buf,
                               self.R, self.I[0], self.I[1], self.I[2])

        return 0;
    
    
