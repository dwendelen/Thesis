
from FUnMapped import FUnMapped

class Float16x16x16E(FUnMapped):
    def getNbOperaties(self, I, R, n):
        return 0
        
    def getDataTransferZonderCache(self, I, R, n):
        return 0
        
    def getDataTransferMetCache(self, I, R, n):
        return 0
    
    def getName(self):
        return 'float16x16x16E'
    
    def initFromE(self, e):
        FUnMapped.initFromFUnMapped(self, e)