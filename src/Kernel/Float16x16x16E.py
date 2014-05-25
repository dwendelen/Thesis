
from Float16x16x16Kernel import *

class Float16x16x16E(Float16x16x16UnmappedKernel):
    def getNbOperaties(self, I, R, n):
        return 0
        
    def getDataTransferZonderCache(self, I, R, n):
        return 0
        
    def getDataTransferMetCache(self, I, R, n):
        return 0
    
    def getName(self):
        return 'float16x16x16E'
    
class Float16x16x16EFactory(Float16x16x16UnmappedKernelFactory):
    def create(self, U, T):
        f = Float16x16x16E(self.contextQueue)
        
        self.initKernelAndCreateCommonBuffers(U, T, f)
        
        return f