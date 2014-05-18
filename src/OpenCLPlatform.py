import pyopencl as cl
import numpy as np
import numpy.linalg as la
from kr import kr
from code import *
from math import *

from Platform import Platform

class OpenCLPlatform (Platform):        
    def init(self):
        devices = cl.get_platforms()[0].get_devices(cl.device_type.GPU)
        context = cl.Context([devices[0]])
        queue = cl.CommandQueue(context)
        
        file = open('../opencl/16x16x16float.cl', 'r')
        
        prg = cl.Program(context, file.read()).build()
            
        self.prg = prg
        self.queue = queue
        self.context = context
    
    def globalSize(self, (i1, i2, i3)):
        return (int(ceil(i1/16.0))*4, int(ceil(i2/16.0))*4, int(ceil(i3/16.0))*4)

    def getTShape(self, (i1, i2, i3)):
        return (int(ceil(i1/16.0))*16, int(ceil(i2/16.0))*16, int(ceil(i3/16.0))*16)

    def createU(self, U):
        elements = int(ceil(U.shape[0]/16.0))*16
        r = np.zeros((elements, U.shape[1]), order='F', dtype=np.float32)
        #r = np.zeros((elements, U.shape[1]), dtype=np.float32)
        r[:U.shape[0],:] = U
        return r

    def f(self):
        U0 = self.createU(self.U[0])
        U1 = self.createU(self.U[1])
        U2 = self.createU(self.U[2])
        
        g = self.globalSize(self.T.shape)
        gs = self.getTShape(self.T.shape)

        T = np.zeros(gs, order='F', dtype=np.float32)
        #T = np.zeros(gs, dtype=np.float32)        


        T[:self.T.shape[0], :self.T.shape[1], :self.T.shape[2]] = self.T        

        mf = cl.mem_flags
        T_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=T)
        U0_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=U0)
        U1_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=U1)
        U2_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=U2)
        l_buf = cl.LocalMemory(64*4)
        sum_buf = cl.Buffer(self.context, mf.WRITE_ONLY, size=4)

        kernel = self.prg.float16x16x16
        kernel.set_scalar_arg_dtypes([None, None, None, None, None,
                np.int32, np.int32, np.int32, np.int32, None])
        kernel(self.queue, g, (4,4,4), T_buf, U0_buf, U1_buf, U2_buf, l_buf,
               self.R, g[0], g[1], g[2], sum_buf)
        
        s = np.zeros((1), dtype = np.float32)
        cl.enqueue_copy(self.queue, s, sum_buf)
        
        return s[0]/2
        
    def g(self):
        return []
