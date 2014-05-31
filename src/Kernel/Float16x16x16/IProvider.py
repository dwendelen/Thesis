import numpy as np

class IProvider:
    I = (None, None, None)
    IBuffer = (None, None, None)
    
    def initIBufferI(self, IBuffer, I):
        self.I = I
        self.IBuffer = IBuffer
    
    def init(self, I):
        self.IBuffer = np.int32(I/4)
        self.I = I
