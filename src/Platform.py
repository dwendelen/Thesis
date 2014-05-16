#import pyopencl as cl
import numpy as np
import numpy.linalg as la
from kr import kr
from code import getM
from code import getNbOfDimensions

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

    def setU(self, U):
        self.U = U
    
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
        context = cl.Context(devices)
        queue = cl.CommandQueue(context)
        
        prg = cl.Program(context, """
            __kernel void sum(__global const float *a,
            __global const float *b, __global float *c)
            {
              int gid = get_global_id(0);
              c[gid] = a[gid] + b[gid];
            }
            """).build()
            
        self.prg = prg
        self.queue = queue
        self.context = context
    
    def f(self):
        a = np.random.rand(50000).astype(np.float32)
        b = np.random.rand(50000).astype(np.float32)
        
        mf = cl.mem_flags
        a_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        dest_buf = cl.Buffer(self.context, mf.WRITE_ONLY, b.nbytes)
        
        self.prg.sum(self.queue, a.shape, None, a_buf, b_buf, dest_buf)
        
        a_plus_b = np.empty_like(a)
        cl.enqueue_copy(self.queue, a_plus_b, dest_buf)
        
        print(la.norm(a_plus_b - (a+b)), la.norm(a_plus_b))
        return 0;
    
    
