from Float16x16x16Kernel import Float16x16x16Kernel
import math

class Float16x16x16E(Float16x16x16Kernel):
    def getNbOperaties(self, I, R, n):
        return 0
        
    def getDataTransferZonderCache(self, I, R, n):
        return 0
        
    def getDataTransferMetCache(self, I, R, n):
        return 0
    
    def getName(self):
        return 'float16x16x16E'