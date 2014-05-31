import numpy as np

class IProvider:
    I = (None, None, None)
    IBuffer = (None, None, None)
    
    def initFromIProvider(self, iProvider):
        self.I = iProvider.I
        self.IBuffer = iProvider.IBuffer
    
    def init(self, I):
        self.IBuffer = np.int32(I)
        self.I = I
