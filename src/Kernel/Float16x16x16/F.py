import math

from FUnMapped import FUnMapped

class F(FUnMapped):
    def getNbOperaties(self, I, R, n):
        return self.getNbWorkGroups(I, R, n) * (9216*R + 12287)
        
    def getDataTransferZonderCache(self, I, R, n):
        return 4*self.getNbWorkGroups(I, R, n) * (16*16*16 + 3 * 16 * R)
        
    def getDataTransferMetCache(self, I, R, n):
        i = math.ceil(I/16.0) * 16
        return (i*i*i + 3*i*R) * 4
    
    def getName(self):
        return 'float16x16x16'

    def getNbWGs(self):
        return (self.I[0]*self.I[1]*self.I[2])/(16*16*16)
