from Kernel import Kernel

class Float3DElement(Kernel):
    def getNbOperaties(self, I, R, n):
        return self.getNbWorkGroups(I, R, n)*(3*R + 1)
    
    def getNbWorkGroups(self, I, R, n):
        return I*I*I
        
    def getDataTransferZonderCache(self, I, R, n):
        return (I*I*I*(1 + 3*R)) * 4
        
    def getDataTransferMetCache(self, I, R, n):
        return (I*I*I + 3*I*R) * 4
    
    def getBasicElements(self, I, R, n):
        return I*I*I*R
    
    def getName(self):
        return 'Float3DElement'