from Buffer import Buffer
import numpy as np

class IBuffer16x16x16(Buffer):
    def __init__(self, I0, I1, I2):
        self.IBuffer0 = np.int32(I0/4)
        self.IBuffer1 = np.int32(I1/4)
        self.IBuffer2 = np.int32(I2/4)

class IBuffer16x16x16Factory():
    def createFromT(self, T):
        if(len(T.shape) != 3):
            raise Exception("Illegal shape.")
        
        return IBuffer16x16x16().init(T.shape[0], T.shape[1], T.shape[2])

    def createFromU(self, U):
        if(len(U.shape) != 3):
            raise Exception("Illegal shape.")
        
        return IBuffer16x16x16().init(U[0].shape[0], U[1].shape[0], U[2].shape[0])