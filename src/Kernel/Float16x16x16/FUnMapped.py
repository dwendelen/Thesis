from TInput import TInput
from FCommon import FCommon

class FUnMapped(FCommon, TInput):
    
    def init(self, T, U):
        TInput.init(self, T)
        FCommon.init(self, U)
        
    def initFromFUnMapped(self, fUnMapped):
        TInput.initFromTInput(self, fUnMapped)
        FCommon.initFromFCommon(self, fUnMapped)
